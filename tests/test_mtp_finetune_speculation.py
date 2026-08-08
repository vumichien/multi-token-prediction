import math

from mtp_finetune.self_speculate import bfloat16_ulp


def test_bfloat16_ulp_matches_the_8_bit_mantissa() -> None:
    # bf16 keeps 8 mantissa bits, so spacing is 2**(exponent - 7).
    assert bfloat16_ulp(13.5) == 0.0625  # exponent 3 -> 2**-4
    assert bfloat16_ulp(1.0) == 2.0**-7
    assert bfloat16_ulp(300.0) == 2.0  # exponent 8 -> 2**1


def test_bfloat16_ulp_is_sign_agnostic_and_zero_safe() -> None:
    assert bfloat16_ulp(-13.5) == bfloat16_ulp(13.5)
    assert bfloat16_ulp(0.0) == 0.0


def test_bfloat16_ulp_is_the_actual_bf16_spacing() -> None:
    # Adding half an ulp to a representable value must round away, adding a
    # whole ulp must land on the next representable value.
    import torch

    value = torch.tensor(13.5, dtype=torch.bfloat16)
    step = bfloat16_ulp(13.5)
    assert float(value.float() + step) != float(value.float())
    nudged = torch.tensor(13.5 + step, dtype=torch.bfloat16)
    assert math.isclose(float(nudged), 13.5 + step, rel_tol=1e-6)
