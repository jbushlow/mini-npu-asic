# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile

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
                with allo.meta_for(Ct) as n:
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


CMD_LOAD_WEIGHTS_BANK_0_V1 = 0
CMD_LOAD_WEIGHTS_BANK_1_V1 = 1
CMD_MATMUL_BANK_0_V1 = 2
CMD_MATMUL_BANK_1_V1 = 3


@df.region()
def MXU_v1[
    Rt, Ct, M, NUM_COMMANDS, DEPTH
](
    I_Packed: "UInt(Rt * 32)[NUM_COMMANDS, M]",
    W_Packed: "UInt(Ct * 32)[2, Rt]",
    C_Packed: "UInt(Ct * 32)[NUM_COMMANDS, M]",
    Command_Opcode: "int32[NUM_COMMANDS]",
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

    @df.kernel(
        mapping=[1],
        args=[I_Packed, Command_Opcode],
    )
    def offchip_loadI_v1(
        I_Packed_in: "UInt(Rt * 32)[NUM_COMMANDS, M]",
        Command_Opcode_in: "int32[NUM_COMMANDS]",
    ):
        for command in range(NUM_COMMANDS):
            opcode: int32 = Command_Opcode_in[command]
            if (
                opcode == CMD_MATMUL_BANK_0_V1
                or opcode == CMD_MATMUL_BANK_1_V1
            ):
                for m in range(M):
                    Global_I.put(I_Packed_in[command, m])

    @df.kernel(
        mapping=[1],
        args=[W_Packed, Command_Opcode],
    )
    def offchip_loadW_v1(
        W_Packed_in: "UInt(Ct * 32)[2, Rt]",
        Command_Opcode_in: "int32[NUM_COMMANDS]",
    ):
        for command in range(NUM_COMMANDS):
            opcode: int32 = Command_Opcode_in[command]
            if opcode == CMD_LOAD_WEIGHTS_BANK_0_V1:
                for r in range(Rt):
                    Global_W.put(W_Packed_in[0, r])
            elif opcode == CMD_LOAD_WEIGHTS_BANK_1_V1:
                for r in range(Rt):
                    Global_W.put(W_Packed_in[1, r])

    @df.kernel(
        mapping=[P0, P1],
        args=[Command_Opcode],
    )
    def gemm_v1(
        Command_Opcode_in: "int32[NUM_COMMANDS]",
    ):
        i, j = df.get_pid()

        with allo.meta_if(i == 0 and j == 0):
            for command in range(NUM_COMMANDS):
                opcode: int32 = Command_Opcode_in[command]
                if (
                    opcode == CMD_LOAD_WEIGHTS_BANK_0_V1
                    or opcode == CMD_LOAD_WEIGHTS_BANK_1_V1
                ):
                    for r in range(Rt):
                        Local_W[1].put(Global_W.get())
                elif (
                    opcode == CMD_MATMUL_BANK_0_V1
                    or opcode == CMD_MATMUL_BANK_1_V1
                ):
                    for m in range(M):
                        Local_I[1].put(Global_I.get())

        with allo.meta_elif(i == P0 - 1 and j == P1 - 1):
            for command in range(NUM_COMMANDS):
                opcode: int32 = Command_Opcode_in[command]
                if (
                    opcode == CMD_MATMUL_BANK_0_V1
                    or opcode == CMD_MATMUL_BANK_1_V1
                ):
                    for m in range(M):
                        packed_c: UInt(Ct * 32) = 0
                        with allo.meta_for(Ct) as n:
                            c: int32 = Local_C[n].get()
                            packed_c[n * 32 : (n + 1) * 32] = c
                        Global_C.put(packed_c)

        with allo.meta_elif(i in {0, P0 - 1} and j in {0, P1 - 1}):
            pass

        with allo.meta_elif(j == 0):
            for command in range(NUM_COMMANDS):
                opcode: int32 = Command_Opcode_in[command]
                if (
                    opcode == CMD_MATMUL_BANK_0_V1
                    or opcode == CMD_MATMUL_BANK_1_V1
                ):
                    for m in range(M):
                        packed_i = Local_I[i].get()
                        fifo_I[i - 1, 0].put(
                            packed_i[32 * (i - 1) : 32 * i]
                        )
                        with allo.meta_if(i < Rt):
                            Local_I[i + 1].put(packed_i)

        with allo.meta_elif(i == 0):
            for command in range(NUM_COMMANDS):
                opcode: int32 = Command_Opcode_in[command]
                if (
                    opcode == CMD_LOAD_WEIGHTS_BANK_0_V1
                    or opcode == CMD_LOAD_WEIGHTS_BANK_1_V1
                ):
                    for r in range(Rt):
                        packed_w = Local_W[j].get()
                        fifo_W[0, j - 1].put(
                            packed_w[32 * (j - 1) : 32 * j]
                        )
                        with allo.meta_if(j < Ct):
                            Local_W[j + 1].put(packed_w)
                elif (
                    opcode == CMD_MATMUL_BANK_0_V1
                    or opcode == CMD_MATMUL_BANK_1_V1
                ):
                    for m in range(M):
                        fifo_P[0, j - 1].put(0)

        with allo.meta_elif(i == P0 - 1):
            for command in range(NUM_COMMANDS):
                opcode: int32 = Command_Opcode_in[command]
                if (
                    opcode == CMD_MATMUL_BANK_0_V1
                    or opcode == CMD_MATMUL_BANK_1_V1
                ):
                    for m in range(M):
                        Local_C[j - 1].put(fifo_P[Rt, j - 1].get())

        with allo.meta_elif(j == P1 - 1):
            pass

        with allo.meta_else():
            weight_0: int32 = 0
            weight_1: int32 = 0

            for command in range(NUM_COMMANDS):
                opcode: int32 = Command_Opcode_in[command]

                if (
                    opcode == CMD_LOAD_WEIGHTS_BANK_0_V1
                    or opcode == CMD_LOAD_WEIGHTS_BANK_1_V1
                ):
                    for r in range(Rt):
                        w: int32 = fifo_W[i - 1, j - 1].get()
                        if r == i - 1:
                            if opcode == CMD_LOAD_WEIGHTS_BANK_0_V1:
                                weight_0 = w
                            else:
                                weight_1 = w
                        with allo.meta_if(i < Rt):
                            fifo_W[i, j - 1].put(w)

                elif (
                    opcode == CMD_MATMUL_BANK_0_V1
                    or opcode == CMD_MATMUL_BANK_1_V1
                ):
                    weight: int32 = weight_0
                    if opcode == CMD_MATMUL_BANK_1_V1:
                        weight = weight_1

                    for m in range(M):
                        a: int32 = fifo_I[i - 1, j - 1].get()
                        p: int32 = fifo_P[i - 1, j - 1].get()
                        next_p: int32 = p + a * weight
                        with allo.meta_if(j < Ct):
                            fifo_I[i - 1, j].put(a)
                        fifo_P[i, j - 1].put(next_p)

    @df.kernel(
        mapping=[1],
        args=[C_Packed, Command_Opcode],
    )
    def offchip_store_v1(
        C_Packed_out: "UInt(Ct * 32)[NUM_COMMANDS, M]",
        Command_Opcode_in: "int32[NUM_COMMANDS]",
    ):
        for command in range(NUM_COMMANDS):
            opcode: int32 = Command_Opcode_in[command]
            if (
                opcode == CMD_MATMUL_BANK_0_V1
                or opcode == CMD_MATMUL_BANK_1_V1
            ):
                for m in range(M):
                    C_Packed_out[command, m] = Global_C.get()


TEST_RT = 4
TEST_CT = 4
TEST_M = 3
TEST_DEPTH = 4
TEST_NUM_COMMANDS_V1 = 4


@df.region()
def MXU_v0_test_top(
    I_Packed: "UInt(TEST_RT * 32)[TEST_M]",
    W_Packed: "UInt(TEST_CT * 32)[TEST_RT]",
    C_Packed: "UInt(TEST_CT * 32)[TEST_M]",
):
    @df.kernel(mapping=[1], args=[I_Packed, W_Packed, C_Packed])
    def wrapper_v0(
        local_I: "UInt(TEST_RT * 32)[TEST_M]",
        local_W: "UInt(TEST_CT * 32)[TEST_RT]",
        local_C: "UInt(TEST_CT * 32)[TEST_M]",
    ):
        MXU_v0[
            TEST_RT,
            TEST_CT,
            TEST_M,
            TEST_DEPTH,
        ](local_I, local_W, local_C)


@df.region()
def MXU_v1_test_top(
    I_Packed: "UInt(TEST_RT * 32)[TEST_NUM_COMMANDS_V1, TEST_M]",
    W_Packed: "UInt(TEST_CT * 32)[2, TEST_RT]",
    C_Packed: "UInt(TEST_CT * 32)[TEST_NUM_COMMANDS_V1, TEST_M]",
    Command_Opcode: "int32[TEST_NUM_COMMANDS_V1]",
):
    @df.kernel(
        mapping=[1],
        args=[I_Packed, W_Packed, C_Packed, Command_Opcode],
    )
    def wrapper_v1(
        local_I: "UInt(TEST_RT * 32)[TEST_NUM_COMMANDS_V1, TEST_M]",
        local_W: "UInt(TEST_CT * 32)[2, TEST_RT]",
        local_C: "UInt(TEST_CT * 32)[TEST_NUM_COMMANDS_V1, TEST_M]",
        local_Command_Opcode: "int32[TEST_NUM_COMMANDS_V1]",
    ):
        MXU_v1[
            TEST_RT,
            TEST_CT,
            TEST_M,
            TEST_NUM_COMMANDS_V1,
            TEST_DEPTH,
        ](
            local_I,
            local_W,
            local_C,
            local_Command_Opcode,
        )


def pack_int32_rows(matrix):
    matrix = np.ascontiguousarray(matrix, dtype=np.int32)
    packed_type = get_np_struct_type(matrix.shape[1] * 32)
    return matrix.reshape(-1).view(packed_type).reshape(matrix.shape[0])


def unpack_int32_rows(packed_matrix, rows, columns):
    return packed_matrix.view(np.int32).reshape(rows, columns).copy()


def pack_int32_tiles(matrix):
    matrix = np.ascontiguousarray(matrix, dtype=np.int32)
    packed_type = get_np_struct_type(matrix.shape[-1] * 32)
    return matrix.reshape(-1).view(packed_type).reshape(matrix.shape[:-1])


def unpack_int32_tiles(packed_matrix, shape):
    return packed_matrix.view(np.int32).reshape(shape).copy()


rng = np.random.default_rng(seed=0)

# =========================================================
# Test 1: v0 weight-stationary MXU
# =========================================================

I_v0 = rng.integers(
    low=-4,
    high=5,
    size=(TEST_M, TEST_RT),
    dtype=np.int32,
)
W_v0 = rng.integers(
    low=-4,
    high=5,
    size=(TEST_RT, TEST_CT),
    dtype=np.int32,
)
C_v0 = np.zeros(
    (TEST_M, TEST_CT),
    dtype=np.int32,
)

expected_C_v0 = I_v0 @ W_v0

I_packed_v0 = pack_int32_rows(I_v0)
W_packed_v0 = pack_int32_rows(W_v0)
C_packed_v0 = pack_int32_rows(C_v0)

print("=" * 60)
print("TEST 1: v0 weight-stationary MXU")
print("=" * 60)

simulator_v0 = df.build(MXU_v0_test_top, target="simulator")
simulator_v0(I_packed_v0, W_packed_v0, C_packed_v0)

C_v0 = unpack_int32_rows(
    C_packed_v0,
    TEST_M,
    TEST_CT,
)

print("I:")
print(I_v0)

print("\nW:")
print(W_v0)

print("\nMXU v0 result:")
print(C_v0)

print("\nNumPy reference:")
print(expected_C_v0)

np.testing.assert_array_equal(C_v0, expected_C_v0)
print("\nTEST 1 PASSED")

# =========================================================
# Test 2: v1 command-controlled two-bank MXU
# =========================================================

Command_Opcode_v1 = np.array(
    [
        CMD_LOAD_WEIGHTS_BANK_0_V1,
        CMD_LOAD_WEIGHTS_BANK_1_V1,
        CMD_MATMUL_BANK_0_V1,
        CMD_MATMUL_BANK_1_V1,
    ],
    dtype=np.int32,
)

I_v1 = np.zeros(
    (TEST_NUM_COMMANDS_V1, TEST_M, TEST_RT),
    dtype=np.int32,
)
I_v1[2] = rng.integers(
    low=-4,
    high=5,
    size=(TEST_M, TEST_RT),
    dtype=np.int32,
)
I_v1[3] = rng.integers(
    low=-4,
    high=5,
    size=(TEST_M, TEST_RT),
    dtype=np.int32,
)

W_v1 = rng.integers(
    low=-4,
    high=5,
    size=(2, TEST_RT, TEST_CT),
    dtype=np.int32,
)

C_v1 = np.zeros(
    (TEST_NUM_COMMANDS_V1, TEST_M, TEST_CT),
    dtype=np.int32,
)
expected_C_v1 = np.zeros_like(C_v1)
expected_C_v1[2] = I_v1[2] @ W_v1[0]
expected_C_v1[3] = I_v1[3] @ W_v1[1]

I_packed_v1 = pack_int32_tiles(I_v1)
W_packed_v1 = pack_int32_tiles(W_v1)
C_packed_v1 = pack_int32_tiles(C_v1)

print("\n")
print("=" * 60)
print("TEST 2: v1 command-controlled two-bank MXU")
print("=" * 60)

simulator_v1 = df.build(MXU_v1_test_top, target="simulator")
simulator_v1(
    I_packed_v1,
    W_packed_v1,
    C_packed_v1,
    Command_Opcode_v1,
)

C_v1 = unpack_int32_tiles(
    C_packed_v1,
    (TEST_NUM_COMMANDS_V1, TEST_M, TEST_CT),
)

print("Commands:")
print(Command_Opcode_v1)

print("\nI for bank 0 matmul:")
print(I_v1[2])

print("\nW loaded into bank 0:")
print(W_v1[0])

print("\nMXU v1 bank 0 result:")
print(C_v1[2])

print("\nNumPy bank 0 reference:")
print(expected_C_v1[2])

print("\nI for bank 1 matmul:")
print(I_v1[3])

print("\nW loaded into bank 1:")
print(W_v1[1])

print("\nMXU v1 bank 1 result:")
print(C_v1[3])

print("\nNumPy bank 1 reference:")
print(expected_C_v1[3])

np.testing.assert_array_equal(C_v1, expected_C_v1)
print("\nTEST 2 PASSED")

print("\n")
print("=" * 60)
print("ALL MXU TESTS PASSED")
print("=" * 60)
