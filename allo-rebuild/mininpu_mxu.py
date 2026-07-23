# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile

import pytest
import allo
from allo.ir.types import int32, Stream, UInt, ConstExpr
from allo.utils import get_np_struct_type
import allo.dataflow as df
import allo.backend.hls as hls
import allo.dsl as dsl
import numpy as np


@df.region()
def MXU_v0[
    Rt, Ct, M, DEPTH
](
    I_Packed: "UInt(Rt * 32)[M]",
    W_Packed: "UInt(Ct * 32)[Rt]",
    C_Packed: "UInt(Ct * 32)[M]",
):
    P0: ConstExpr[int32] = Rt + 2
    P1: ConstExpr[int32] = Ct + 2

    Global_I: Stream[UInt(Rt * 32), DEPTH]
    Global_W: Stream[UInt(Ct * 32), DEPTH]
    Global_C: Stream[UInt(Ct * 32), DEPTH]

    Local_I: Stream[UInt(Rt * 32), DEPTH][P0 - 1]
    Local_W: Stream[UInt(Ct * 32), DEPTH][P1 - 1]

    Local_C: Stream[int32, DEPTH][Ct]

    fifo_I: Stream[int32, DEPTH][Rt, Ct]
    fifo_W: Stream[int32, DEPTH][Rt, Ct]
    fifo_P: Stream[int32, DEPTH][Rt + 1, Ct]

    @df.kernel(mapping=[1], args=[I_Packed])
    def offchip_loadI_v0(I_Packed_in: "UInt(Rt * 32)[M]"):
        for m in range(M):
            Global_I.put(I_Packed_in[m])

    @df.kernel(mapping=[1], args=[W_Packed])
    def offchip_loadW_v0(W_Packed_in: "UInt(Ct * 32)[Rt]"):
        for r in range(Rt):
            Global_W.put(W_Packed_in[r])

    @df.kernel(mapping=[P0, P1])
    def gemm_v0():
        i, j = df.get_pid()
        # peripheral kernels
        with allo.meta_if(i == 0 and j == 0):
            for r in range(Rt):
                Local_W[1].put(Global_W.get())
            for m in range(M):
                Local_I[1].put(Global_I.get())

        with allo.meta_elif(i == P0 - 1 and j == P1 - 1):
            for m in range(M):
                packed_c: UInt(Ct * 32) = 0
                for n in range(Ct):
                    c: int32 = Local_C[n].get()
                    packed_c[n * 32 : (n + 1) * 32] = c
                Global_C.put(packed_c)

        with allo.meta_elif(i in {0, P0 - 1} and j in {0, P1 - 1}):
            pass

        with allo.meta_elif(j == 0):
            # i > 0, the first column
            for m in range(M):
                a = Local_I[i].get()
                # unpack data
                fifo_I[i - 1, 0].put(a[32 * (i - 1) : 32 * i])
                with allo.meta_if(i < Rt):
                    Local_I[i + 1].put(a)

        with allo.meta_elif(i == 0):
            # j > 0, the first row
            for r in range(Rt):
                w = Local_W[j].get()
                fifo_W[0, j - 1].put(w[32 * (j - 1) : 32 * j])
                with allo.meta_if(j < Ct):
                    Local_W[j + 1].put(w)
            for m in range(M):
                fifo_P[0, j - 1].put(0)

        with allo.meta_elif(i == P0 - 1):
            for m in range(M):
                Local_C[j - 1].put(fifo_P[Rt, j - 1].get())

        with allo.meta_elif(j == P1 - 1):
            pass

        # main body
        with allo.meta_else():
            weight: int32 = 0
            for r in range(Rt):
                w: int32 = fifo_W[i - 1, j - 1].get()
                if r == i - 1:
                    weight = w
                with allo.meta_if(i < Rt):
                    fifo_W[i, j - 1].put(w)

            for m in range(M):
                a: int32 = fifo_I[i - 1, j - 1].get()
                p: int32 = fifo_P[i - 1, j - 1].get()
                next_p: int32 = p + a * weight
                with allo.meta_if(j < Ct):
                    fifo_I[i - 1, j].put(a)
                fifo_P[i, j - 1].put(next_p)

    @df.kernel(mapping=[1], args=[C_Packed])
    def offchip_store_v0(C_Packed_out: "UInt(Ct * 32)[M]"):
        for m in range(M):
            C_Packed_out[m] = Global_C.get()


TEST_RT_V0 = 4
TEST_CT_V0 = 4
TEST_M_V0 = 3
TEST_DEPTH_V0 = 4


@df.region()
def MXU_v0_test_top(
    I_Packed: "UInt(TEST_RT_V0 * 32)[TEST_M_V0]",
    W_Packed: "UInt(TEST_CT_V0 * 32)[TEST_RT_V0]",
    C_Packed: "UInt(TEST_CT_V0 * 32)[TEST_M_V0]",
):
    @df.kernel(mapping=[1], args=[I_Packed, W_Packed, C_Packed])
    def wrapper_v0(
        local_I: "UInt(TEST_RT_V0 * 32)[TEST_M_V0]",
        local_W: "UInt(TEST_CT_V0 * 32)[TEST_RT_V0]",
        local_C: "UInt(TEST_CT_V0 * 32)[TEST_M_V0]",
    ):
        MXU_v0[
            TEST_RT_V0,
            TEST_CT_V0,
            TEST_M_V0,
            TEST_DEPTH_V0,
        ](local_I, local_W, local_C)


def pack_int32_rows_v0(matrix):
    matrix = np.ascontiguousarray(matrix, dtype=np.int32)
    packed_type = get_np_struct_type(matrix.shape[1] * 32)
    return matrix.reshape(-1).view(packed_type).reshape(matrix.shape[0])


def unpack_int32_rows_v0(packed_matrix, rows, columns):
    return packed_matrix.view(np.int32).reshape(rows, columns).copy()


@pytest.mark.parametrize(
    "I, W",
    [
        (
            np.array(
                [
                    [1, 2, 3, 4],
                    [-5, 6, -7, 8],
                    [9, -10, 11, -12],
                ],
                dtype=np.int32,
            ),
            np.eye(TEST_RT_V0, dtype=np.int32),
        ),
        (
            np.array(
                [
                    [1, -2, 3, 0],
                    [4, 5, -1, 2],
                    [-3, 1, 2, 6],
                ],
                dtype=np.int32,
            ),
            np.array(
                [
                    [2, 1, 0, -1],
                    [-3, 4, 2, 1],
                    [5, -2, 3, 0],
                    [1, 6, -4, 2],
                ],
                dtype=np.int32,
            ),
        ),
    ],
)
def test_MXU_v0(I, W):
    expected_C = I @ W

    I_packed = pack_int32_rows_v0(I)
    W_packed = pack_int32_rows_v0(W)
    C_packed = pack_int32_rows_v0(
        np.zeros((TEST_M_V0, TEST_CT_V0), dtype=np.int32)
    )

    simulator_v0 = df.build(MXU_v0_test_top, target="simulator")
    simulator_v0(I_packed, W_packed, C_packed)

    actual_C = unpack_int32_rows_v0(
        C_packed,
        TEST_M_V0,
        TEST_CT_V0,
    )

    print("\nI:")
    print(I)
    print("\nW:")
    print(W)
    print("\nMXU v0 result:")
    print(actual_C)
    print("\nNumPy reference:")
    print(expected_C)

    np.testing.assert_array_equal(actual_C, expected_C)


if __name__ == "__main__":
    test_cases_v0 = [
        (
            np.array(
                [
                    [1, 2, 3, 4],
                    [-5, 6, -7, 8],
                    [9, -10, 11, -12],
                ],
                dtype=np.int32,
            ),
            np.eye(TEST_RT_V0, dtype=np.int32),
        ),
        (
            np.array(
                [
                    [1, -2, 3, 0],
                    [4, 5, -1, 2],
                    [-3, 1, 2, 6],
                ],
                dtype=np.int32,
            ),
            np.array(
                [
                    [2, 1, 0, -1],
                    [-3, 4, 2, 1],
                    [5, -2, 3, 0],
                    [1, 6, -4, 2],
                ],
                dtype=np.int32,
            ),
        ),
    ]

    for test_I_v0, test_W_v0 in test_cases_v0:
        test_MXU_v0(test_I_v0, test_W_v0)

    print("\nALL MXU v0 TESTS PASSED")
