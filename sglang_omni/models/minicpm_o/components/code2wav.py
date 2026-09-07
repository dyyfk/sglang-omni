# SPDX-License-Identifier: Apache-2.0
"""Code2Wav component for MiniCPM-o.

Wraps stepaudio2's ``Token2wav`` (the remote code's ``init_tts`` dependency)
to turn s3tokenizer codec tokens into a 24 kHz waveform. The vocoder assets
live in the checkpoint directory under ``assets/token2wav``; the default
prompt (speaker reference) wav is ``assets/HT_ref_audio.wav`` when present,
matching the remote demo's default voice.
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np
import torch
import torch.nn as nn

from sglang_omni.models.weight_loader import resolve_model_path

logger = logging.getLogger(__name__)

OUTPUT_SAMPLE_RATE = 24000

# Streaming constants from the checkpoint's reference streaming loop
# (``streaming_generate`` in modeling_minicpmo.py): the token buffer is
# seeded with 3 silence tokens, each flow call consumes one 25-token chunk
# plus 3 lookahead tokens, and the buffer advances by 25 so the lookahead
# tokens are re-fed as the head of the next chunk.
STREAM_CHUNK_SIZE = 25
STREAM_PRE_LOOKAHEAD = 3
STREAM_SILENCE_TOKEN = 4218
STREAM_SILENCE_PREFIX = 3


def _clone_tree(value):
    """Deep-clone the nested tensor containers Token2wav uses as caches."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_tree(item) for item in value)
    return value


class MiniCPMOCode2Wav(nn.Module):
    """stepaudio2 Token2wav wrapper: codec tokens → float32 waveform."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        float16: bool = False,
        n_timesteps: int = 10,
        prompt_wav: str | None = None,
    ) -> None:
        super().__init__()
        del device  # Token2wav manages its own device placement (cuda).
        try:
            from stepaudio2 import Token2wav
        except ImportError as exc:
            raise ImportError(
                "MiniCPM-o audio output requires stepaudio2; install via "
                "pip install minicpmo-utils[all]"
            ) from exc

        model_dir = str(resolve_model_path(model_path))
        asset_dir = os.path.join(model_dir, "assets", "token2wav")
        if not os.path.isdir(asset_dir):
            raise FileNotFoundError(
                f"token2wav assets not found at {asset_dir}; copy the "
                "checkpoint's assets/token2wav directory next to the weights"
            )
        self.token2wav = Token2wav(asset_dir, float16=float16, n_timesteps=n_timesteps)

        if prompt_wav is None:
            default_wav = os.path.join(model_dir, "assets", "HT_ref_audio.wav")
            prompt_wav = default_wav if os.path.isfile(default_wav) else None
        self._prompt_wav = prompt_wav
        # Note (ruoyu): Token2wav's prompt cache is shared mutable state, so
        # build it once here instead of lazily on the first request, and guard
        # the fallback (and the stream-base build below) with a lock.
        self._prompt_cache_lock = threading.Lock()
        if self._prompt_wav is not None:
            self.token2wav.cache = self.token2wav._prepare_prompt(self._prompt_wav)
        self._stream_base: tuple[object, dict] | None = None

    @torch.inference_mode()
    def forward(
        self,
        *,
        codec_tokens: torch.Tensor,
        prompt_wav: str | None = None,
        **_: object,
    ) -> dict[str, object]:
        """Vocode one utterance.

        Args:
            codec_tokens: ``(N,)`` s3tokenizer codes (EOS already stripped).
            prompt_wav: optional path to a 16 kHz speaker-reference wav;
                falls back to the component default.

        Returns:
            ``waveform``: ``(samples,)`` float32 at 24 kHz; ``sample_rate``.
        """
        tokens = codec_tokens.reshape(-1).tolist()
        if not tokens:
            return {
                "waveform": np.zeros(0, dtype=np.float32),
                "sample_rate": OUTPUT_SAMPLE_RATE,
            }
        waveform = self._vocode(tokens, prompt_wav or self._prompt_wav)
        return {"waveform": waveform, "sample_rate": OUTPUT_SAMPLE_RATE}

    def _vocode(self, tokens: list[int], prompt_wav: str | None) -> np.ndarray:
        """``Token2wav.__call__`` minus its final ``torchaudio.save`` — newer
        torchaudio (torchcodec backend) cannot encode into ``BytesIO``, and we
        want the raw waveform anyway."""
        t2w = self.token2wav
        if t2w.cache is None:
            with self._prompt_cache_lock:
                if t2w.cache is None:
                    t2w.cache = t2w._prepare_prompt(prompt_wav)
        (
            prompt_speech_tokens,
            prompt_speech_tokens_lens,
            spk_emb,
            prompt_mels,
            prompt_mels_lens,
        ) = t2w.cache

        speech_tokens = torch.tensor([tokens], dtype=torch.int32, device="cuda")
        speech_tokens_lens = torch.tensor(
            [speech_tokens.shape[1]], dtype=torch.int32, device="cuda"
        )
        with torch.amp.autocast(
            "cuda", dtype=torch.float16 if t2w.float16 else torch.float32
        ):
            mel = t2w.flow.inference(
                speech_tokens,
                speech_tokens_lens,
                prompt_speech_tokens,
                prompt_speech_tokens_lens,
                prompt_mels,
                prompt_mels_lens,
                spk_emb,
                t2w.n_timesteps,
            )
        wav, _ = t2w.hift(speech_feat=mel)
        return wav.reshape(-1).float().cpu().numpy()

    # ------------------------------------------------------------------
    # Chunked streaming (Token2wav.stream with per-request caches)
    # ------------------------------------------------------------------

    def _ensure_stream_base(self) -> tuple[object, dict]:
        """Flow/hift caches primed on the speaker prompt, computed once.

        Replicates ``Token2wav.set_stream_cache`` (right-pads the prompt codes
        with 3 silence tokens) but returns the caches instead of mutating
        Token2wav instance state, so concurrent requests can each stream from
        their own clone.
        """
        if self._stream_base is not None:
            return self._stream_base
        with self._prompt_cache_lock:
            if self._stream_base is not None:
                return self._stream_base
            t2w = self.token2wav
            if t2w.cache is None:
                if self._prompt_wav is None:
                    raise RuntimeError(
                        "MiniCPM-o streaming vocode requires a speaker prompt "
                        "wav (assets/HT_ref_audio.wav) to prime the flow cache"
                    )
                t2w.cache = t2w._prepare_prompt(self._prompt_wav)
            prompt_speech_tokens, _, spk_emb, prompt_mels, _ = t2w.cache
            right_pad = torch.full(
                (1, STREAM_SILENCE_PREFIX),
                STREAM_SILENCE_TOKEN,
                device=prompt_speech_tokens.device,
                dtype=prompt_speech_tokens.dtype,
            )
            flow_cache = t2w.flow.setup_cache(
                torch.cat([prompt_speech_tokens, right_pad], dim=1),
                prompt_mels,
                spk_emb,
                n_timesteps=getattr(t2w, "n_timesteps", 10),
            )
            hift_cache = {
                "mel": torch.zeros(
                    1, prompt_mels.shape[2], 0, device=prompt_mels.device
                ),
                "source": torch.zeros(1, 1, 0, device=prompt_mels.device),
                "speech": torch.zeros(1, 0, device=prompt_mels.device),
            }
            self._stream_base = (flow_cache, hift_cache)
        return self._stream_base

    def new_stream_state(self) -> dict:
        """Per-request streaming caches, cloned from the shared prompt base."""
        flow_base, hift_base = self._ensure_stream_base()
        return {"flow": _clone_tree(flow_base), "hift": _clone_tree(hift_base)}

    @torch.inference_mode()
    def stream_step(
        self, stream_state: dict, tokens: list[int], *, last_chunk: bool
    ) -> np.ndarray:
        """One ``Token2wav.stream`` call against per-request caches.

        Mirrors the shipped ``stream(..., return_waveform=True)``: flow
        ``inference_chunk`` with attention-cache truncation, hift with the
        rolling mel/source/speech cache, Hamming cross-fade against the
        previous chunk's tail, and first-chunk silence left-padding.
        """
        from stepaudio2.token2wav import fade_in_out

        t2w = self.token2wav
        _, _, spk_emb, prompt_mels, _ = t2w.cache
        token_tensor = torch.tensor(
            [tokens], dtype=torch.int32, device=prompt_mels.device
        )
        with torch.amp.autocast(
            "cuda", dtype=torch.float16 if t2w.float16 else torch.float32
        ):
            chunk_mel, stream_state["flow"] = t2w.flow.inference_chunk(
                token=token_tensor,
                spk=spk_emb,
                cache=stream_state["flow"],
                last_chunk=last_chunk,
                n_timesteps=getattr(t2w, "n_timesteps", 10),
            )
        flow_cache = stream_state["flow"]
        cache_limit = prompt_mels.shape[1] + 100
        if flow_cache["estimator_att_cache"].shape[4] > cache_limit:
            flow_cache["estimator_att_cache"] = torch.cat(
                [
                    flow_cache["estimator_att_cache"][
                        :, :, :, :, : prompt_mels.shape[1]
                    ],
                    flow_cache["estimator_att_cache"][:, :, :, :, -100:],
                ],
                dim=4,
            )
        if (
            "conformer_att_cache" in flow_cache
            and flow_cache["conformer_att_cache"].shape[3] > cache_limit
        ):
            flow_cache["conformer_att_cache"] = torch.cat(
                [
                    flow_cache["conformer_att_cache"][
                        :, :, :, : prompt_mels.shape[1], :
                    ],
                    flow_cache["conformer_att_cache"][:, :, :, -100:, :],
                ],
                dim=3,
            )

        hift_cache = stream_state["hift"]
        mel = torch.concat([hift_cache["mel"], chunk_mel], dim=2)
        speech, source = t2w.hift(mel, hift_cache["source"])
        is_first_chunk = hift_cache["speech"].shape[-1] == 0
        if not is_first_chunk:
            speech = fade_in_out(speech, hift_cache["speech"], t2w.speech_window)
        stream_state["hift"] = {
            "mel": mel[..., -t2w.mel_cache_len :].clone().detach(),
            "source": source[:, :, -t2w.source_cache_len :].clone().detach(),
            "speech": speech[:, -t2w.source_cache_len :].clone().detach(),
        }
        if not last_chunk:
            if is_first_chunk:
                # First chunk: hold back the cross-fade tail and pad the head
                # with silence so chunk lengths stay uniform (shipped behavior).
                silence = torch.zeros(1, t2w.source_cache_len, device=speech.device)
                speech = torch.cat([silence, speech[:, : -t2w.source_cache_len]], dim=1)
            else:
                speech = speech[:, : -t2w.source_cache_len]
        return speech.reshape(-1).float().cpu().numpy()
