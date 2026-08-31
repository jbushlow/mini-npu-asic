import struct

import numpy as np

NUM_VECTORS = 16


def fp32_bits(value):
    return struct.unpack(">I", struct.pack(">f", float(value)))[0]


def bf16_bits(value):
    return fp32_bits(value) >> 16


def bf16_value(bits):
    return struct.unpack(">f", struct.pack(">I", bits << 16))[0]


def vector_word(weight, activation, psum):
    """Return {expected_fp32, psum_in_fp32, activation_bf16, weight_bf16}."""
    weight_bits = bf16_bits(weight)
    activation_bits = bf16_bits(activation)
    expected = np.float32(
        np.float32(bf16_value(activation_bits))
        * np.float32(bf16_value(weight_bits))
        + np.float32(psum)
    )
    return (
        f"{fp32_bits(expected):08x}"
        f"{fp32_bits(psum):08x}"
        f"{activation_bits:04x}"
        f"{weight_bits:04x}"
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

def main():
    if len(CASES) != NUM_VECTORS:
        raise RuntimeError(f"Expected {NUM_VECTORS} cases, got {len(CASES)}")

    with open("test_vectors.txt", "w", encoding="ascii") as f:
        for case in CASES:
            f.write(vector_word(*case) + "\n")


if __name__ == "__main__":
    main()
