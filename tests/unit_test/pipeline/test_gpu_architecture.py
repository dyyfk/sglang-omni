# SPDX-License-Identifier: Apache-2.0

from sglang_omni.utils.gpu_compat import (
    architecture_supports_flash_attention_3,
    architecture_supports_flash_attention_4,
    gpu_architecture_for_sm,
    preferred_sglang_attention_backend,
)


def test_gpu_architectures_do_not_conflate_blackwell_variants() -> None:
    assert gpu_architecture_for_sm(89) == "ada"
    assert gpu_architecture_for_sm(90) == "hopper"
    assert gpu_architecture_for_sm(100) == "blackwell-datacenter"
    assert gpu_architecture_for_sm(120) == "blackwell-consumer"
    assert architecture_supports_flash_attention_3(90)
    assert not architecture_supports_flash_attention_3(89)
    assert architecture_supports_flash_attention_4(100)
    assert not architecture_supports_flash_attention_4(120)
    assert preferred_sglang_attention_backend(89) == "flashinfer"
    assert preferred_sglang_attention_backend(90) == "fa3"
