import struct

import numpy as np

NUM_VECTORS = 16


def fp16_bits(value):
    return int(np.float16(value).view(np.uint16))


def fp32_bits(value):
    return struct.unpack(">I", struct.pack(">f", float(value)))[0]


def vector_word(weight, activation, psum):
    """Return {expected_fp32, psum_in_fp32, activation_fp16, weight_fp16}."""
    weight_f16 = np.float16(weight)
    activation_f16 = np.float16(activation)
    expected = np.float32(np.float32(activation_f16) * np.float32(weight_f16) + np.float32(psum))
    return (
        f"{fp32_bits(expected):08x}"
        f"{fp32_bits(psum):08x}"
        f"{fp16_bits(activation):04x}"
        f"{fp16_bits(weight):04x}"
    )


CASES = [
    (1.0, 1.0, 0.0),
    (2.0, 3.0, 0.0),
    (-2.0, 3.0, 0.0),
    (2.0, -3.0, 0.0),
    (-2.0, -3.0, 0.0),
    (0.5, 4.0, 1.0),
    (1.5, 2.0, -1.0),
    (-1.5, 2.0, 5.0),
    (0.0, 7.0, 9.0),
    (7.0, 0.0, -3.0),
    (8.0, 8.0, 1.0),
    (-8.0, 8.0, 1.0),
    (0.25, 0.5, 2.0),
    (16.0, 0.25, -4.0),
    (3.0, 5.0, 10.0),
    (-3.0, 5.0, 10.0),
]

if len(CASES) != NUM_VECTORS:
    raise RuntimeError(f"Expected {NUM_VECTORS} cases, got {len(CASES)}")

with open("test_vectors.txt", "w", encoding="ascii") as f:
    for case in CASES:
        f.write(vector_word(*case) + "\n")
