# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import allo
import allo.dataflow as df
import numpy as np
from allo.ir.types import int32, Stream, Stateful


# Configuration

LANES = 4
IMEM_SIZE = 16
INSTR_W = 8
NUM_INPUT_IDS = 16
NUM_VECTOR_IDS = 16
NUM_EXT_INPUTS = 16
NUM_WEIGHT_TILES = 8
NUM_EXT_OUTPUTS = 16
STREAM_DEPTH = 16


# Host control

CTRL_LOAD_INSTRUCTION = 0
CTRL_RUN_PROGRAM = 1


# MXU operations
#
# MXU_LOAD_WEIGHTS_0/1:
#   F0 = external weight-tile index
# MXU_LOAD_INPUT:
#   F0 = input ID, F1 = external input-vector index
# MXU_GO_0/1:
#   F0 = input ID; the same ID is attached to the MXU result

MXU_NOP = 0
MXU_LOAD_WEIGHTS_0 = 1
MXU_LOAD_WEIGHTS_1 = 2
MXU_LOAD_INPUT = 3
MXU_GO_0 = 4
MXU_GO_1 = 5


# VPU operations
#
# VPU_CAPTURE_MXU:
#   pop one MXU result and store it at the result's input ID
# VPU_VADD:
#   vector_buffer[DST] = vector_buffer[SRC0] + vector_buffer[SRC1]
# VPU_RELU:
#   vector_buffer[DST] = relu(vector_buffer[SRC0])
# VPU_STORE:
#   ext_outputs[DST] = vector_buffer[SRC0]

VPU_NOP = 0
VPU_CAPTURE_MXU = 1
VPU_VADD = 2
VPU_RELU = 3
VPU_STORE = 4


# VLIW instruction fields

MXU_OP = 0
MXU_F0 = 1
MXU_F1 = 2
MXU_F2 = 3
VPU_OP = 4
VPU_DST = 5
VPU_SRC0 = 6
VPU_SRC1 = 7


@df.region()
def mininpu_v2(
    ctrl: int32[1],
    instr_addr: int32[1],
    instr_in: int32[INSTR_W],
    ext_inputs: int32[NUM_EXT_INPUTS, LANES],
    ext_weights: int32[NUM_WEIGHT_TILES, LANES, LANES],
    ext_outputs: int32[NUM_EXT_OUTPUTS, LANES],
):
    # Sequencer-to-engine control

    mxu_frontend_mode_s: Stream[int32, 1]
    mxu_broadcast_mode_s: Stream[int32, 1]
    vpu_mode_s: Stream[int32, 1]

    mxu_command_s: Stream[int32[4], STREAM_DEPTH]
    vpu_command_s: Stream[int32[4], STREAM_DEPTH]

    # Sequencer-to-MXU data

    input_push_s: Stream[int32[LANES], STREAM_DEPTH]
    weight_push_s: Stream[int32[LANES], STREAM_DEPTH]

    # MXU internal communication

    mxu_execute_command_s: Stream[int32[4], STREAM_DEPTH]
    mxu_input_s: Stream[int32[LANES], STREAM_DEPTH]
    mxu_input_id_s: Stream[int32, STREAM_DEPTH]

    mxu_pe_mode_s: Stream[int32, 1][LANES + 2, LANES + 2]
    mxu_pe_command_s: Stream[int32[4], STREAM_DEPTH][LANES + 2, LANES + 2]

    Local_I: Stream[int32[LANES], STREAM_DEPTH][LANES + 1]
    Local_W: Stream[int32[LANES], STREAM_DEPTH][LANES + 1]
    Local_C: Stream[int32, STREAM_DEPTH][LANES]

    fifo_I: Stream[int32, STREAM_DEPTH][LANES, LANES]
    fifo_W: Stream[int32, STREAM_DEPTH][LANES, LANES]
    fifo_P: Stream[int32, STREAM_DEPTH][LANES + 1, LANES]

    mxu_inflight_id_s: Stream[int32, STREAM_DEPTH]

    # MXU-to-VPU results

    mxu_result_s: Stream[int32[LANES], STREAM_DEPTH]
    mxu_result_id_s: Stream[int32, STREAM_DEPTH]

    # Sequencer

    @df.kernel(
        mapping=[1],
        args=[ctrl, instr_addr, instr_in, ext_inputs, ext_weights],
    )
    def sequencer(
        ctrl_in: int32[1],
        instr_addr_in: int32[1],
        instr_in_local: int32[INSTR_W],
        ext_inputs_local: int32[NUM_EXT_INPUTS, LANES],
        ext_weights_local: int32[NUM_WEIGHT_TILES, LANES, LANES],
    ):
        imem: int32[IMEM_SIZE * INSTR_W] @ Stateful = 0

        mode: int32 = ctrl_in[0]
        mxu_frontend_mode_s.put(mode)
        mxu_broadcast_mode_s.put(mode)
        vpu_mode_s.put(mode)

        if mode == CTRL_LOAD_INSTRUCTION:
            base: int32 = instr_addr_in[0] * INSTR_W
            for field in range(INSTR_W):
                imem[base + field] = instr_in_local[field]

        elif mode == CTRL_RUN_PROGRAM:
            for pc in range(IMEM_SIZE):
                base: int32 = pc * INSTR_W

                mxu_command: int32[4] = 0
                mxu_command[0] = imem[base + MXU_OP]
                mxu_command[1] = imem[base + MXU_F0]
                mxu_command[2] = imem[base + MXU_F1]
                mxu_command[3] = imem[base + MXU_F2]
                mxu_command_s.put(mxu_command)

                vpu_command: int32[4] = 0
                vpu_command[0] = imem[base + VPU_OP]
                vpu_command[1] = imem[base + VPU_DST]
                vpu_command[2] = imem[base + VPU_SRC0]
                vpu_command[3] = imem[base + VPU_SRC1]
                vpu_command_s.put(vpu_command)

                mxu_op: int32 = mxu_command[0]
                mxu_f0: int32 = mxu_command[1]
                mxu_f1: int32 = mxu_command[2]

                if mxu_op == MXU_LOAD_INPUT:
                    input_value: int32[LANES] = 0
                    for lane in range(LANES):
                        input_value[lane] = ext_inputs_local[mxu_f1, lane]
                    input_push_s.put(input_value)

                elif (
                    mxu_op == MXU_LOAD_WEIGHTS_0
                    or mxu_op == MXU_LOAD_WEIGHTS_1
                ):
                    for row in range(LANES):
                        weight_row: int32[LANES] = 0
                        for lane in range(LANES):
                            weight_row[lane] = ext_weights_local[
                                mxu_f0,
                                row,
                                lane,
                            ]
                        weight_push_s.put(weight_row)

    # MXU input buffer and tag forwarding

    @df.kernel(mapping=[1])
    def mxu_input_buffer():
        input_buffer: int32[NUM_INPUT_IDS, LANES] = 0

        mode: int32 = mxu_frontend_mode_s.get()
        if mode == CTRL_RUN_PROGRAM:
            for pc in range(IMEM_SIZE):
                command: int32[4] = mxu_command_s.get()
                opcode: int32 = command[0]
                input_id: int32 = command[1]

                if opcode == MXU_LOAD_INPUT:
                    input_value: int32[LANES] = input_push_s.get()
                    for lane in range(LANES):
                        input_buffer[input_id, lane] = input_value[lane]

                elif opcode == MXU_GO_0 or opcode == MXU_GO_1:
                    input_value: int32[LANES] = 0
                    for lane in range(LANES):
                        input_value[lane] = input_buffer[input_id, lane]
                    mxu_input_s.put(input_value)
                    mxu_input_id_s.put(input_id)

                mxu_execute_command_s.put(command)

    # MXU command broadcast

    @df.kernel(mapping=[1])
    def mxu_command_broadcast():
        mode: int32 = mxu_broadcast_mode_s.get()

        with allo.meta_for(LANES + 2) as i:
            with allo.meta_for(LANES + 2) as j:
                mxu_pe_mode_s[i, j].put(mode)

        if mode == CTRL_RUN_PROGRAM:
            for pc in range(IMEM_SIZE):
                command: int32[4] = mxu_execute_command_s.get()
                with allo.meta_for(LANES + 2) as i:
                    with allo.meta_for(LANES + 2) as j:
                        mxu_pe_command_s[i, j].put(command)

    # Weight-stationary systolic MXU

    @df.kernel(mapping=[LANES + 2, LANES + 2])
    def mxu_systolic_array():
        i, j = df.get_pid()

        mode: int32 = mxu_pe_mode_s[i, j].get()
        if mode == CTRL_RUN_PROGRAM:
            # Top-left input and weight injection
            with allo.meta_if(i == 0 and j == 0):
                for pc in range(IMEM_SIZE):
                    command: int32[4] = mxu_pe_command_s[i, j].get()
                    opcode: int32 = command[0]

                    if (
                        opcode == MXU_LOAD_WEIGHTS_0
                        or opcode == MXU_LOAD_WEIGHTS_1
                    ):
                        for row in range(LANES):
                            Local_W[1].put(weight_push_s.get())

                    elif opcode == MXU_GO_0 or opcode == MXU_GO_1:
                        Local_I[1].put(mxu_input_s.get())
                        mxu_inflight_id_s.put(mxu_input_id_s.get())

            # Bottom-right result collection
            with allo.meta_elif(i == LANES + 1 and j == LANES + 1):
                for pc in range(IMEM_SIZE):
                    command: int32[4] = mxu_pe_command_s[i, j].get()
                    opcode: int32 = command[0]

                    if opcode == MXU_GO_0 or opcode == MXU_GO_1:
                        result: int32[LANES] = 0
                        with allo.meta_for(LANES) as lane:
                            result[lane] = Local_C[lane].get()
                        mxu_result_s.put(result)
                        mxu_result_id_s.put(mxu_inflight_id_s.get())

            # Unused peripheral corners
            with allo.meta_elif(
                i in {0, LANES + 1} and j in {0, LANES + 1}
            ):
                for pc in range(IMEM_SIZE):
                    unused_command: int32[4] = mxu_pe_command_s[i, j].get()

            # Left-edge input distribution
            with allo.meta_elif(j == 0):
                for pc in range(IMEM_SIZE):
                    command: int32[4] = mxu_pe_command_s[i, j].get()
                    opcode: int32 = command[0]

                    if opcode == MXU_GO_0 or opcode == MXU_GO_1:
                        input_value: int32[LANES] = Local_I[i].get()
                        fifo_I[i - 1, 0].put(input_value[i - 1])
                        with allo.meta_if(i < LANES):
                            Local_I[i + 1].put(input_value)

            # Top-edge weight distribution and partial-sum initialization
            with allo.meta_elif(i == 0):
                for pc in range(IMEM_SIZE):
                    command: int32[4] = mxu_pe_command_s[i, j].get()
                    opcode: int32 = command[0]

                    if (
                        opcode == MXU_LOAD_WEIGHTS_0
                        or opcode == MXU_LOAD_WEIGHTS_1
                    ):
                        for row in range(LANES):
                            weight_row: int32[LANES] = Local_W[j].get()
                            fifo_W[0, j - 1].put(weight_row[j - 1])
                            with allo.meta_if(j < LANES):
                                Local_W[j + 1].put(weight_row)

                    elif opcode == MXU_GO_0 or opcode == MXU_GO_1:
                        fifo_P[0, j - 1].put(0)

            # Bottom-edge partial-sum drain
            with allo.meta_elif(i == LANES + 1):
                for pc in range(IMEM_SIZE):
                    command: int32[4] = mxu_pe_command_s[i, j].get()
                    opcode: int32 = command[0]

                    if opcode == MXU_GO_0 or opcode == MXU_GO_1:
                        Local_C[j - 1].put(fifo_P[LANES, j - 1].get())

            # Unused right edge
            with allo.meta_elif(j == LANES + 1):
                for pc in range(IMEM_SIZE):
                    unused_command: int32[4] = mxu_pe_command_s[i, j].get()

            # Compute PEs
            with allo.meta_else():
                weight_0: int32 = 0
                weight_1: int32 = 0

                for pc in range(IMEM_SIZE):
                    command: int32[4] = mxu_pe_command_s[i, j].get()
                    opcode: int32 = command[0]

                    if (
                        opcode == MXU_LOAD_WEIGHTS_0
                        or opcode == MXU_LOAD_WEIGHTS_1
                    ):
                        for row in range(LANES):
                            weight: int32 = fifo_W[i - 1, j - 1].get()
                            if row == i - 1:
                                if opcode == MXU_LOAD_WEIGHTS_0:
                                    weight_0 = weight
                                else:
                                    weight_1 = weight
                            with allo.meta_if(i < LANES):
                                fifo_W[i, j - 1].put(weight)

                    elif opcode == MXU_GO_0 or opcode == MXU_GO_1:
                        input_value: int32 = fifo_I[i - 1, j - 1].get()
                        partial_sum: int32 = fifo_P[i - 1, j - 1].get()

                        weight: int32 = weight_0
                        if opcode == MXU_GO_1:
                            weight = weight_1

                        next_partial_sum: int32 = (
                            partial_sum + input_value * weight
                        )

                        with allo.meta_if(j < LANES):
                            fifo_I[i - 1, j].put(input_value)
                        fifo_P[i, j - 1].put(next_partial_sum)

    # Four-lane SIMD VPU and vector buffer

    @df.kernel(mapping=[1], args=[ext_outputs])
    def vpu(ext_outputs_local: int32[NUM_EXT_OUTPUTS, LANES]):
        vector_buffer: int32[NUM_VECTOR_IDS, LANES] = 0

        mode: int32 = vpu_mode_s.get()
        if mode == CTRL_RUN_PROGRAM:
            for pc in range(IMEM_SIZE):
                command: int32[4] = vpu_command_s.get()
                opcode: int32 = command[0]
                dst: int32 = command[1]
                src0: int32 = command[2]
                src1: int32 = command[3]

                if opcode == VPU_CAPTURE_MXU:
                    result: int32[LANES] = mxu_result_s.get()
                    result_id: int32 = mxu_result_id_s.get()
                    for lane in range(LANES):
                        vector_buffer[result_id, lane] = result[lane]

                elif opcode == VPU_VADD:
                    for lane in range(LANES):
                        vector_buffer[dst, lane] = (
                            vector_buffer[src0, lane]
                            + vector_buffer[src1, lane]
                        )

                elif opcode == VPU_RELU:
                    for lane in range(LANES):
                        value: int32 = vector_buffer[src0, lane]
                        if value < 0:
                            value = 0
                        vector_buffer[dst, lane] = value

                elif opcode == VPU_STORE:
                    for lane in range(LANES):
                        ext_outputs_local[dst, lane] = vector_buffer[src0, lane]


def build(project, target, mode, configs):
    """Build using the standard allo-asic-compilation node contract."""
    project = Path(project).resolve()
    module = df.build(
        mininpu_v2,
        target=target,
        mode=mode,
        project=project,
        wrap_io=False,
        configs=configs,
    )
    module()


def workload():
    """Create a short program exercising the sequencer, MXU, and VPU.

    The program computes::

        output[0] = relu(input[0] @ weight[0] + input[1] @ weight[1])

    Unused instruction slots remain all-zero NOPs. Values are deliberately
    small enough that every intermediate fits comfortably in int32.
    """
    instructions = np.zeros((IMEM_SIZE, INSTR_W), dtype=np.int32)

    # Load two independent stationary weight banks.
    instructions[0, MXU_OP] = MXU_LOAD_WEIGHTS_0
    instructions[0, MXU_F0] = 0
    instructions[1, MXU_OP] = MXU_LOAD_WEIGHTS_1
    instructions[1, MXU_F0] = 1

    # Buffer input 0, execute it with bank 0, and capture under input ID 0.
    instructions[2, MXU_OP] = MXU_LOAD_INPUT
    instructions[2, MXU_F0] = 0
    instructions[2, MXU_F1] = 0
    instructions[3, MXU_OP] = MXU_GO_0
    instructions[3, MXU_F0] = 0
    instructions[3, VPU_OP] = VPU_CAPTURE_MXU

    # Buffer input 1, execute it with bank 1, and capture under input ID 1.
    instructions[4, MXU_OP] = MXU_LOAD_INPUT
    instructions[4, MXU_F0] = 1
    instructions[4, MXU_F1] = 1
    instructions[5, MXU_OP] = MXU_GO_1
    instructions[5, MXU_F0] = 1
    instructions[5, VPU_OP] = VPU_CAPTURE_MXU

    # Add the two captured vectors, apply ReLU, and store external output 0.
    instructions[6, VPU_OP] = VPU_VADD
    instructions[6, VPU_DST] = 2
    instructions[6, VPU_SRC0] = 0
    instructions[6, VPU_SRC1] = 1
    instructions[7, VPU_OP] = VPU_RELU
    instructions[7, VPU_DST] = 3
    instructions[7, VPU_SRC0] = 2
    instructions[8, VPU_OP] = VPU_STORE
    instructions[8, VPU_DST] = 0
    instructions[8, VPU_SRC0] = 3

    ext_inputs = np.zeros((NUM_EXT_INPUTS, LANES), dtype=np.int32)
    ext_inputs[0] = np.array([2, -1, 3, 1], dtype=np.int32)
    ext_inputs[1] = np.array([-2, 4, 1, 3], dtype=np.int32)

    ext_weights = np.zeros(
        (NUM_WEIGHT_TILES, LANES, LANES), dtype=np.int32
    )
    ext_weights[0] = np.array(
        [
            [1, 2, 0, -1],
            [0, 1, 3, 2],
            [2, -1, 1, 0],
            [-2, 0, 1, 2],
        ],
        dtype=np.int32,
    )
    ext_weights[1] = np.array(
        [
            [2, 0, -1, 1],
            [1, -2, 0, 3],
            [0, 1, 2, -1],
            [3, 1, 0, 2],
        ],
        dtype=np.int32,
    )

    bank_0_result = ext_inputs[0] @ ext_weights[0]
    bank_1_result = ext_inputs[1] @ ext_weights[1]
    expected_outputs = np.zeros(
        (NUM_EXT_OUTPUTS, LANES), dtype=np.int32
    )
    expected_outputs[0] = np.maximum(bank_0_result + bank_1_result, 0)

    return {
        "instructions": instructions,
        "ext_inputs": ext_inputs,
        "ext_weights": ext_weights,
        "expected_outputs": expected_outputs,
    }


def run_workload():
    """Load the example program through the host interface and execute it."""
    transaction = workload()
    instructions = transaction["instructions"]
    ext_inputs = transaction["ext_inputs"]
    ext_weights = transaction["ext_weights"]
    ext_outputs = np.zeros((NUM_EXT_OUTPUTS, LANES), dtype=np.int32)

    simulator = df.build(mininpu_v2, target="simulator")

    # Instruction memory is explicitly Stateful, so each host load call
    # updates one slot and retains all previously written slots.
    for pc in range(IMEM_SIZE):
        ctrl = np.array([CTRL_LOAD_INSTRUCTION], dtype=np.int32)
        instr_addr = np.array([pc], dtype=np.int32)
        simulator(
            ctrl,
            instr_addr,
            instructions[pc],
            ext_inputs,
            ext_weights,
            ext_outputs,
        )

    ctrl = np.array([CTRL_RUN_PROGRAM], dtype=np.int32)
    instr_addr = np.zeros(1, dtype=np.int32)
    empty_instruction = np.zeros(INSTR_W, dtype=np.int32)
    simulator(
        ctrl,
        instr_addr,
        empty_instruction,
        ext_inputs,
        ext_weights,
        ext_outputs,
    )

    np.testing.assert_array_equal(
        ext_outputs, transaction["expected_outputs"]
    )
    print("MiniNPU v2 sequencer/MXU/VPU workload passed")
    print("Output 0:", ext_outputs[0])


if __name__ == "__main__":
    run_workload()
