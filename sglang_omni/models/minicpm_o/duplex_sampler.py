# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o duplex token sampling over a batch of thinker sessions."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang_omni.models.minicpm_o.special_tokens import MiniCPMOSpecialTokenIds
from sglang_omni.models.minicpm_o.thinker_state import MiniCPMOThinkerSessionState


def build_forbidden_token_index(
    special_tokens: MiniCPMOSpecialTokenIds, vocab_size: int, device: torch.device
) -> torch.Tensor:
    """Rows the second-stage sample never picks, resolved once instead of per step."""
    forbidden_token_ids = sorted(
        token_id
        for token_id in {special_tokens.chunk_eos, *special_tokens.forbidden}
        if token_id < vocab_size
    )
    return torch.tensor(forbidden_token_ids, dtype=torch.long, device=device)


def filter_top_k_top_p(
    logits: torch.Tensor, *, top_k: int, top_p: float
) -> torch.Tensor:
    """Mask every row of a (rows, vocab) batch outside its top-k and its nucleus."""
    filtered_logits = logits.clone()
    if 0 < top_k < filtered_logits.shape[-1]:
        thresholds = torch.topk(filtered_logits, top_k, dim=-1).values[:, -1:]
        filtered_logits[filtered_logits < thresholds] = -torch.inf
    else:
        pass
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            filtered_logits, descending=True, dim=-1
        )
        cumulative_probabilities = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )
        should_remove = cumulative_probabilities > top_p
        should_remove[:, 1:] = should_remove[:, :-1].clone()
        should_remove[:, 0] = False
        filtered_logits.scatter_(
            -1, sorted_indices, sorted_logits.masked_fill(should_remove, -torch.inf)
        )
    else:
        pass
    return filtered_logits


def sample_rows(logits: torch.Tensor, greedy_flags: list[bool]) -> torch.Tensor:
    """One token id per row: the argmax for greedy rows, a softmax draw for the rest."""
    if all(greedy_flags):
        return torch.argmax(logits, dim=-1)
    elif not any(greedy_flags):
        return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
    else:
        token_ids = torch.argmax(logits, dim=-1)
        sampled_rows = torch.tensor(
            [row for row, is_greedy in enumerate(greedy_flags) if not is_greedy],
            dtype=torch.long,
            device=logits.device,
        )
        token_ids[sampled_rows] = torch.multinomial(
            F.softmax(logits[sampled_rows], dim=-1), 1
        ).squeeze(-1)
        return token_ids


def duplex_sample_batch(
    logits: torch.Tensor,
    thinker_states: list[MiniCPMOThinkerSessionState],
    *,
    special_tokens: MiniCPMOSpecialTokenIds,
    forbidden_token_index: torch.Tensor,
    generation_steps: list[int],
    listen_forced_flags: list[bool],
) -> list[int]:
    """Apply the two-stage unit sampler to every row with two host transfers per step."""
    device = logits.device
    token_ids: list[int | None] = [None] * len(thinker_states)
    candidate_rows: list[int] = []
    for row, thinker_state in enumerate(thinker_states):
        sampling = thinker_state.sampling
        if generation_steps[row] >= sampling.max_new_tokens_per_unit - 1:
            token_ids[row] = special_tokens.chunk_eos
        elif generation_steps[row] == 0 and listen_forced_flags[row]:
            token_ids[row] = special_tokens.listen
        else:
            candidate_rows.append(row)
    if not candidate_rows:
        return token_ids
    else:
        pass

    candidate_logits = logits[
        torch.tensor(candidate_rows, dtype=torch.long, device=device)
    ].float()
    # note (Junnan Li): The first sample must use the unscaled model distribution.
    first_token_ids = sample_rows(
        candidate_logits,
        [thinker_states[row].sampling.greedy for row in candidate_rows],
    ).tolist()
    second_positions: list[int] = []
    for position, row in enumerate(candidate_rows):
        if first_token_ids[position] == special_tokens.chunk_eos:
            token_ids[row] = special_tokens.chunk_eos
        else:
            second_positions.append(position)
    if not second_positions:
        return token_ids
    else:
        pass

    second_states = [
        thinker_states[candidate_rows[position]] for position in second_positions
    ]
    second_logits = candidate_logits[
        torch.tensor(second_positions, dtype=torch.long, device=device)
    ]
    second_logits[:, forbidden_token_index] = -torch.inf
    apply_repetition_penalty(second_logits, second_states)

    listen_scales = [state.sampling.listen_prob_scale for state in second_states]
    if any(scale != 1.0 for scale in listen_scales):
        second_logits[:, special_tokens.listen] *= torch.tensor(
            listen_scales, dtype=torch.float32, device=device
        )
    else:
        pass

    candidate_token_ids = sample_candidates(second_logits, second_states).tolist()
    for position, candidate_token_id in zip(
        second_positions, candidate_token_ids, strict=True
    ):
        thinker_state = thinker_states[candidate_rows[position]]
        sampling = thinker_state.sampling
        # note (Junnan Li): History retains controls before the mid-turn listen rewrite.
        history = thinker_state.generated_history
        history.append(candidate_token_id)
        del history[: -sampling.repetition_window_size]
        if (
            candidate_token_id == special_tokens.listen
            and not thinker_state.is_turn_ended
        ):
            candidate_token_id = special_tokens.tts_bos
        else:
            pass
        if candidate_token_id == special_tokens.turn_eos:
            thinker_state.is_turn_ended = True
        elif candidate_token_id not in special_tokens.chunk_terminators:
            thinker_state.is_turn_ended = False
        else:
            pass
        token_ids[candidate_rows[position]] = candidate_token_id
    return token_ids


def apply_repetition_penalty(
    logits: torch.Tensor, thinker_states: list[MiniCPMOThinkerSessionState]
) -> None:
    """Divide each row's recently generated token logits by that session's penalty, in place."""
    windows: list[list[int]] = []
    for thinker_state in thinker_states:
        sampling = thinker_state.sampling
        history = thinker_state.generated_history
        if sampling.repetition_penalty != 1.0 and history:
            windows.append(sorted(set(history[-sampling.repetition_window_size :])))
        else:
            windows.append([])
    width = max(len(window) for window in windows)
    if width == 0:
        return
    else:
        pass
    vocab_size = logits.shape[-1]
    # note (Junnan Li): Matches the checkpoint sampler, which ignores the logit sign.
    penalties = torch.tensor(
        [[state.sampling.repetition_penalty] for state in thinker_states],
        dtype=torch.float32,
        device=logits.device,
    )
    # A spare vocabulary column absorbs the padding of shorter windows.
    padded_index = torch.tensor(
        [window + [vocab_size] * (width - len(window)) for window in windows],
        dtype=torch.long,
        device=logits.device,
    )
    widened_logits = torch.cat([logits, logits.new_zeros(logits.shape[0], 1)], dim=-1)
    widened_logits.scatter_(
        -1, padded_index, widened_logits.gather(-1, padded_index) / penalties
    )
    logits.copy_(widened_logits[:, :vocab_size])


def sample_candidates(
    logits: torch.Tensor, thinker_states: list[MiniCPMOThinkerSessionState]
) -> torch.Tensor:
    """Second-stage token per row: argmax for greedy rows, filtered draws grouped by settings."""
    token_ids = torch.argmax(logits, dim=-1)
    sampled_groups: dict[tuple[float, int, float], list[int]] = {}
    for row, thinker_state in enumerate(thinker_states):
        sampling = thinker_state.sampling
        if sampling.greedy or sampling.temperature <= 0:
            pass
        else:
            sampled_groups.setdefault(
                (sampling.temperature, sampling.top_k, sampling.top_p), []
            ).append(row)
    for (temperature, top_k, top_p), rows in sampled_groups.items():
        row_index = torch.tensor(rows, dtype=torch.long, device=logits.device)
        filtered_logits = filter_top_k_top_p(
            logits[row_index] / temperature, top_k=top_k, top_p=top_p
        )
        token_ids[row_index] = torch.multinomial(
            F.softmax(filtered_logits, dim=-1), 1
        ).squeeze(-1)
    return token_ids
