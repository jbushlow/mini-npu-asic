# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import allo
import allo.dataflow as df
import numpy as np
from allo.ir.types import float16, int8, int16, Stream, Stateful


# Configuration

LANES = 5
STREAM_DEPTH = 4
MXU_STREAM_DEPTH = 2

IMEM_SIZE = 16
INSTR_W = 8
NUM_INPUT_SLOTS = 16
NUM_VECTOR_SLOTS = 16
NUM_EXT_INPUTS = 16
NUM_WEIGHT_TILES = 8
NUM_EXT_OUTPUTS = 16


# Host control

CTRL_LOAD_INSTRUCTION = 0
CTRL_RUN_PROGRAM = 1


# MXU operations:
# MXU_LOAD_WEIGHTS_0/1: F0 = external weight-tile index
# MXU_LOAD_INPUT: F0 = destination input-buffer slot, F1 = external input index
# MXU_GO_0/1: F0 = source input-buffer slot

MXU_NOP = 0
MXU_LOAD_WEIGHTS_0 = 1
MXU_LOAD_WEIGHTS_1 = 2
MXU_LOAD_INPUT = 3
MXU_GO_0 = 4
MXU_GO_1 = 5


# VPU operations:
# VPU_CAPTURE_MXU: pop next MXU result into vector_buffer[DST]
# VPU_VADD: vector_buffer[DST] = vector_buffer[SRC0] + vector_buffer[SRC1]
# VPU_RELU: vector_buffer[DST] = relu(vector_buffer[SRC0])
# VPU_STORE: ext_outputs[DST] = vector_buffer[SRC0]

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
    ctrl: int8[1],
    instr_addr: int8[1],
    instr_in: int8[INSTR_W],
    ext_inputs: float16[NUM_EXT_INPUTS * LANES],
    ext_weights: float16[NUM_WEIGHT_TILES * LANES * LANES],
    ext_outputs: float16[NUM_EXT_OUTPUTS * LANES],
):
    # Sequencer-to-engine control

    mxu_frontend_mode_s: Stream[int8, 1]
    mxu_broadcast_mode_s: Stream[int8, 1]
    vpu_mode_s: Stream[int8, 1]

    mxu_command_s: Stream[int8[4], MXU_STREAM_DEPTH]
    vpu_command_s: Stream[int8[4], STREAM_DEPTH]

    # Sequencer-to-MXU data

    input_push_s: Stream[float16[LANES], MXU_STREAM_DEPTH]
    weight_push_s: Stream[float16[LANES], MXU_STREAM_DEPTH]

    # MXU internal communication

    mxu_execute_command_s: Stream[int8[4], MXU_STREAM_DEPTH]
    mxu_input_s: Stream[float16[LANES], MXU_STREAM_DEPTH]

    mxu_pe_mode_s: Stream[int8, 1][LANES + 2, LANES + 2]
    mxu_pe_command_s: Stream[int8[4], MXU_STREAM_DEPTH][LANES + 2, LANES + 2]

    Local_I: Stream[float16[LANES], MXU_STREAM_DEPTH][LANES + 1]
    Local_W: Stream[float16[LANES], MXU_STREAM_DEPTH][LANES + 1]
    Local_C: Stream[float16, MXU_STREAM_DEPTH][LANES]

    fifo_I: Stream[float16, MXU_STREAM_DEPTH][LANES, LANES]
    fifo_W: Stream[float16, MXU_STREAM_DEPTH][LANES, LANES]
    fifo_P: Stream[float16, MXU_STREAM_DEPTH][LANES + 1, LANES]

    # MXU-to-VPU results are untagged and consumed in production order. The
    # matching VPU_CAPTURE_MXU instruction selects the destination with DST.

    mxu_result_s: Stream[float16[LANES], MXU_STREAM_DEPTH]

    # VPU dispatch, identical lanes, and output collection

    vpu_lane_mode_s: Stream[int8, 1][LANES]
    vpu_store_mode_s: Stream[int8, 1]
    vpu_lane_command_s: Stream[int8[4], STREAM_DEPTH][LANES]
    vpu_store_command_s: Stream[int8[4], STREAM_DEPTH]
    vpu_lane_result_s: Stream[float16, STREAM_DEPTH][LANES]
    vpu_lane_store_s: Stream[float16, STREAM_DEPTH][LANES]

    # Sequencer

    @df.kernel(
        mapping=[1],
        args=[ctrl, instr_addr, instr_in, ext_inputs, ext_weights],
    )
    def sequencer(
        ctrl_in: int8[1],
        instr_addr_in: int8[1],
        instr_in_local: int8[INSTR_W],
        ext_inputs_local: float16[NUM_EXT_INPUTS * LANES],
        ext_weights_local: float16[NUM_WEIGHT_TILES * LANES * LANES],
    ):
        imem: int8[IMEM_SIZE * INSTR_W] @ Stateful = 0

        mode: int8 = ctrl_in[0]
        mxu_frontend_mode_s.put(mode)
        mxu_broadcast_mode_s.put(mode)
        vpu_mode_s.put(mode)

        if mode == CTRL_LOAD_INSTRUCTION:
            base: int8 = instr_addr_in[0] * INSTR_W
            for field in range(INSTR_W):
                imem[base + field] = instr_in_local[field]

        elif mode == CTRL_RUN_PROGRAM:
            for pc in range(IMEM_SIZE):
                base: int8 = pc * INSTR_W

                mxu_command: int8[4] = 0
                mxu_command[0] = imem[base + MXU_OP]
                mxu_command[1] = imem[base + MXU_F0]
                mxu_command[2] = imem[base + MXU_F1]
                mxu_command[3] = imem[base + MXU_F2]
                mxu_command_s.put(mxu_command)

                vpu_command: int8[4] = 0
                vpu_command[0] = imem[base + VPU_OP]
                vpu_command[1] = imem[base + VPU_DST]
                vpu_command[2] = imem[base + VPU_SRC0]
                vpu_command[3] = imem[base + VPU_SRC1]
                vpu_command_s.put(vpu_command)

                mxu_op: int8 = mxu_command[0]
                mxu_f0: int8 = mxu_command[1]
                mxu_f1: int8 = mxu_command[2]

                if mxu_op == MXU_LOAD_INPUT:
                    input_value: float16[LANES] = 0
                    for lane in range(LANES):
                        input_value[lane] = ext_inputs_local[
                            mxu_f1 * LANES + lane
                        ]
                    input_push_s.put(input_value)

                elif (
                    mxu_op == MXU_LOAD_WEIGHTS_0
                    or mxu_op == MXU_LOAD_WEIGHTS_1
                ):
                    for row in range(LANES):
                        weight_row: float16[LANES] = 0
                        for lane in range(LANES):
                            weight_tile: int16 = mxu_f0
                            weight_index: int16 = (
                                (weight_tile * LANES + row) * LANES + lane
                            )
                            weight_row[lane] = ext_weights_local[
                                weight_index
                            ]
                        weight_push_s.put(weight_row)

    # MXU input buffering and command forwarding

    @df.kernel(mapping=[1])
    def mxu_input_buffer():
        input_buffer: float16[NUM_INPUT_SLOTS, LANES] = 0

        mode: int8 = mxu_frontend_mode_s.get()
        if mode == CTRL_RUN_PROGRAM:
            for pc in range(IMEM_SIZE):
                command: int8[4] = mxu_command_s.get()
                opcode: int8 = command[0]
                input_slot: int8 = command[1]

                if opcode == MXU_LOAD_INPUT:
                    input_value: float16[LANES] = input_push_s.get()
                    for lane in range(LANES):
                        input_buffer[input_slot, lane] = input_value[lane]

                elif opcode == MXU_GO_0 or opcode == MXU_GO_1:
                    input_value: float16[LANES] = 0
                    for lane in range(LANES):
                        input_value[lane] = input_buffer[input_slot, lane]
                    mxu_input_s.put(input_value)

                mxu_execute_command_s.put(command)

    # MXU command broadcast

    @df.kernel(mapping=[1])
    def mxu_command_broadcast():
        mode: int8 = mxu_broadcast_mode_s.get()

        with allo.meta_for(LANES + 2) as i:
            with allo.meta_for(LANES + 2) as j:
                mxu_pe_mode_s[i, j].put(mode)

        if mode == CTRL_RUN_PROGRAM:
            for pc in range(IMEM_SIZE):
                command: int8[4] = mxu_execute_command_s.get()
                with allo.meta_for(LANES + 2) as i:
                    with allo.meta_for(LANES + 2) as j:
                        mxu_pe_command_s[i, j].put(command)

    # Weight-stationary systolic MXU

    @df.kernel(mapping=[LANES + 2, LANES + 2])
    def mxu_systolic_array():
        i, j = df.get_pid()

        mode: int8 = mxu_pe_mode_s[i, j].get()
        if mode == CTRL_RUN_PROGRAM:
            # Top-left input and weight injection
            with allo.meta_if(i == 0 and j == 0):
                for pc in range(IMEM_SIZE):
                    command: int8[4] = mxu_pe_command_s[i, j].get()
                    opcode: int8 = command[0]

                    if (
                        opcode == MXU_LOAD_WEIGHTS_0
                        or opcode == MXU_LOAD_WEIGHTS_1
                    ):
                        for row in range(LANES):
                            Local_W[1].put(weight_push_s.get())

                    elif opcode == MXU_GO_0 or opcode == MXU_GO_1:
                        Local_I[1].put(mxu_input_s.get())

            # Bottom-right result collection
            with allo.meta_elif(i == LANES + 1 and j == LANES + 1):
                for pc in range(IMEM_SIZE):
                    command: int8[4] = mxu_pe_command_s[i, j].get()
                    opcode: int8 = command[0]

                    if opcode == MXU_GO_0 or opcode == MXU_GO_1:
                        result: float16[LANES] = 0
                        with allo.meta_for(LANES) as lane:
                            result[lane] = Local_C[lane].get()
                        mxu_result_s.put(result)

            # Unused peripheral corners
            with allo.meta_elif(
                i in {0, LANES + 1} and j in {0, LANES + 1}
            ):
                for pc in range(IMEM_SIZE):
                    unused_command: int8[4] = mxu_pe_command_s[i, j].get()

            # Left-edge input distribution
            with allo.meta_elif(j == 0):
                for pc in range(IMEM_SIZE):
                    command: int8[4] = mxu_pe_command_s[i, j].get()
                    opcode: int8 = command[0]

                    if opcode == MXU_GO_0 or opcode == MXU_GO_1:
                        input_value: float16[LANES] = Local_I[i].get()
                        fifo_I[i - 1, 0].put(input_value[i - 1])
                        with allo.meta_if(i < LANES):
                            Local_I[i + 1].put(input_value)

            # Top-edge weight distribution and partial-sum initialization
            with allo.meta_elif(i == 0):
                for pc in range(IMEM_SIZE):
                    command: int8[4] = mxu_pe_command_s[i, j].get()
                    opcode: int8 = command[0]

                    if (
                        opcode == MXU_LOAD_WEIGHTS_0
                        or opcode == MXU_LOAD_WEIGHTS_1
                    ):
                        for row in range(LANES):
                            weight_row: float16[LANES] = Local_W[j].get()
                            fifo_W[0, j - 1].put(weight_row[j - 1])
                            with allo.meta_if(j < LANES):
                                Local_W[j + 1].put(weight_row)

                    elif opcode == MXU_GO_0 or opcode == MXU_GO_1:
                        fifo_P[0, j - 1].put(0)

            # Bottom-edge partial-sum drain
            with allo.meta_elif(i == LANES + 1):
                for pc in range(IMEM_SIZE):
                    command: int8[4] = mxu_pe_command_s[i, j].get()
                    opcode: int8 = command[0]

                    if opcode == MXU_GO_0 or opcode == MXU_GO_1:
                        Local_C[j - 1].put(fifo_P[LANES, j - 1].get())

            # Unused right edge
            with allo.meta_elif(j == LANES + 1):
                for pc in range(IMEM_SIZE):
                    unused_command: int8[4] = mxu_pe_command_s[i, j].get()

            # Compute PEs
            with allo.meta_else():
                weight_0: float16 = 0
                weight_1: float16 = 0

                for pc in range(IMEM_SIZE):
                    command: int8[4] = mxu_pe_command_s[i, j].get()
                    opcode: int8 = command[0]

                    if (
                        opcode == MXU_LOAD_WEIGHTS_0
                        or opcode == MXU_LOAD_WEIGHTS_1
                    ):
                        for row in range(LANES):
                            weight: float16 = fifo_W[i - 1, j - 1].get()
                            if row == i - 1:
                                if opcode == MXU_LOAD_WEIGHTS_0:
                                    weight_0 = weight
                                else:
                                    weight_1 = weight
                            with allo.meta_if(i < LANES):
                                fifo_W[i, j - 1].put(weight)

                    elif opcode == MXU_GO_0 or opcode == MXU_GO_1:
                        input_value: float16 = fifo_I[i - 1, j - 1].get()
                        partial_sum: float16 = fifo_P[i - 1, j - 1].get()

                        weight: float16 = weight_0
                        if opcode == MXU_GO_1:
                            weight = weight_1

                        next_partial_sum: float16 = (
                            partial_sum + input_value * weight
                        )

                        with allo.meta_if(j < LANES):
                            fifo_I[i - 1, j].put(input_value)
                        fifo_P[i, j - 1].put(next_partial_sum)

    # SIMD VPU dispatch. This is the sole consumer of ordered, untagged MXU
    # results and broadcasts scalar lane data plus identical commands to the
    # mapped lanes. Each lane uses the command's DST field for capture.

    @df.kernel(mapping=[1])
    def vpu_dispatch():
        mode: int8 = vpu_mode_s.get()
        with allo.meta_for(LANES) as lane:
            vpu_lane_mode_s[lane].put(mode)
        vpu_store_mode_s.put(mode)

        if mode == CTRL_RUN_PROGRAM:
            for pc in range(IMEM_SIZE):
                command: int8[4] = vpu_command_s.get()
                opcode: int8 = command[0]

                with allo.meta_for(LANES) as lane:
                    vpu_lane_command_s[lane].put(command)
                vpu_store_command_s.put(command)

                if opcode == VPU_CAPTURE_MXU:
                    result: float16[LANES] = mxu_result_s.get()
                    with allo.meta_for(LANES) as lane:
                        vpu_lane_result_s[lane].put(result[lane])

    # Each mapped VPU lane owns one scalar slice of the vector buffer. The
    # kernel body has no PID-dependent control, so all lane implementations are
    # structurally equivalent and suitable for one reused macro class.

    @df.kernel(mapping=[LANES])
    def vpu_lane():
        lane = df.get_pid()
        vector_buffer: float16[NUM_VECTOR_SLOTS] = 0

        mode: int8 = vpu_lane_mode_s[lane].get()
        if mode == CTRL_RUN_PROGRAM:
            for pc in range(IMEM_SIZE):
                command: int8[4] = vpu_lane_command_s[lane].get()
                opcode: int8 = command[0]
                dst: int8 = command[1]
                src0: int8 = command[2]
                src1: int8 = command[3]

                if opcode == VPU_CAPTURE_MXU:
                    result: float16 = vpu_lane_result_s[lane].get()
                    vector_buffer[dst] = result

                elif opcode == VPU_VADD:
                    vector_buffer[dst] = (
                        vector_buffer[src0] + vector_buffer[src1]
                    )

                elif opcode == VPU_RELU:
                    value: float16 = vector_buffer[src0]
                    if value < 0:
                        value = 0
                    vector_buffer[dst] = value

                elif opcode == VPU_STORE:
                    vpu_lane_store_s[lane].put(vector_buffer[src0])

    # Keep external-memory addressing out of vpu_lane so the lane PID does not
    # become a lane-specific address constant inside the reusable macro.

    @df.kernel(mapping=[1], args=[ext_outputs])
    def vpu_store(ext_outputs_local: float16[NUM_EXT_OUTPUTS * LANES]):
        mode: int8 = vpu_store_mode_s.get()
        if mode == CTRL_RUN_PROGRAM:
            for pc in range(IMEM_SIZE):
                command: int8[4] = vpu_store_command_s.get()
                opcode: int8 = command[0]
                dst: int8 = command[1]

                if opcode == VPU_STORE:
                    with allo.meta_for(LANES) as lane:
                        ext_outputs_local[dst * LANES + lane] = (
                            vpu_lane_store_s[lane].get()
                        )


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
    small enough to remain exactly representable throughout this fp16 example.
    """
    instructions = np.zeros((IMEM_SIZE, INSTR_W), dtype=np.int8)

    # Load two independent stationary weight banks.
    instructions[0, MXU_OP] = MXU_LOAD_WEIGHTS_0
    instructions[0, MXU_F0] = 0
    instructions[1, MXU_OP] = MXU_LOAD_WEIGHTS_1
    instructions[1, MXU_F0] = 1

    # Buffer input 0, execute it with bank 0, and capture in vector slot 0.
    instructions[2, MXU_OP] = MXU_LOAD_INPUT
    instructions[2, MXU_F0] = 0
    instructions[2, MXU_F1] = 0
    instructions[3, MXU_OP] = MXU_GO_0
    instructions[3, MXU_F0] = 0
    instructions[3, VPU_OP] = VPU_CAPTURE_MXU
    instructions[3, VPU_DST] = 0

    # Buffer input 1, execute it with bank 1, and capture in vector slot 1.
    instructions[4, MXU_OP] = MXU_LOAD_INPUT
    instructions[4, MXU_F0] = 1
    instructions[4, MXU_F1] = 1
    instructions[5, MXU_OP] = MXU_GO_1
    instructions[5, MXU_F0] = 1
    instructions[5, VPU_OP] = VPU_CAPTURE_MXU
    instructions[5, VPU_DST] = 1

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

    ext_inputs = np.zeros((NUM_EXT_INPUTS, LANES), dtype=np.float16)
    ext_inputs[0] = np.array([2, -1, 3, 1, 2], dtype=np.float16)
    ext_inputs[1] = np.array([-2, 4, 1, 3, -1], dtype=np.float16)

    ext_weights = np.zeros(
        (NUM_WEIGHT_TILES, LANES, LANES), dtype=np.float16
    )
    ext_weights[0] = np.array(
        [
            [1, 2, 0, -1, 1],
            [0, 1, 3, 2, -2],
            [2, -1, 1, 0, 1],
            [-2, 0, 1, 2, 3],
            [1, -2, 2, 0, 1],
        ],
        dtype=np.float16,
    )
    ext_weights[1] = np.array(
        [
            [2, 0, -1, 1, 2],
            [1, -2, 0, 3, -1],
            [0, 1, 2, -1, 1],
            [3, 1, 0, 2, -2],
            [-1, 2, 1, 0, 3],
        ],
        dtype=np.float16,
    )

    bank_0_result = ext_inputs[0] @ ext_weights[0]
    bank_1_result = ext_inputs[1] @ ext_weights[1]
    expected_outputs = np.zeros(
        (NUM_EXT_OUTPUTS, LANES), dtype=np.float16
    )
    expected_outputs[0] = np.maximum(bank_0_result + bank_1_result, 0)

    return {
        "instructions": instructions,
        "ext_inputs": ext_inputs.reshape(-1),
        "ext_weights": ext_weights.reshape(-1),
        "expected_outputs": expected_outputs.reshape(-1),
    }


def run_workload():
    """Load the example program through the host interface and execute it."""
    transaction = workload()
    instructions = transaction["instructions"]
    ext_inputs = transaction["ext_inputs"]
    ext_weights = transaction["ext_weights"]
    ext_outputs = np.zeros(NUM_EXT_OUTPUTS * LANES, dtype=np.float16)

    simulator = df.build(mininpu_v2, target="simulator")

    # Instruction memory is explicitly Stateful, so each host load call
    # updates one slot and retains all previously written slots.
    for pc in range(IMEM_SIZE):
        ctrl = np.array([CTRL_LOAD_INSTRUCTION], dtype=np.int8)
        instr_addr = np.array([pc], dtype=np.int8)
        simulator(
            ctrl,
            instr_addr,
            instructions[pc],
            ext_inputs,
            ext_weights,
            ext_outputs,
        )

    ctrl = np.array([CTRL_RUN_PROGRAM], dtype=np.int8)
    instr_addr = np.zeros(1, dtype=np.int8)
    empty_instruction = np.zeros(INSTR_W, dtype=np.int8)
    simulator(
        ctrl,
        instr_addr,
        empty_instruction,
        ext_inputs,
        ext_weights,
        ext_outputs,
    )

    np.testing.assert_allclose(
        ext_outputs, transaction["expected_outputs"], rtol=1e-3, atol=1e-3
    )
    print("MiniNPU v2 sequencer/MXU/VPU workload passed")
    print("Output 0:", ext_outputs.reshape(NUM_EXT_OUTPUTS, LANES)[0])


def run_vitis(project, frequency, device):
    """Compile MiniNPU v2 with Vitis HLS C synthesis."""
    project = Path(project).resolve()
    module = df.build(
        mininpu_v2,
        target="vitis_hls",
        mode="csyn",
        project=project,
        wrap_io=False,
        configs={
            "frequency": frequency,
            "device": device,
        },
    )
    print(f"Generated Vitis project: {project}")
    print(f"Generated HLS source: {project / 'kernel.cpp'}")
    module()
    print("Vitis HLS synthesis completed")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate MiniNPU v2 or synthesize it with Vitis HLS."
    )
    parser.add_argument(
        "--vitis",
        action="store_true",
        help="run Vitis HLS C synthesis instead of the simulator",
    )
    parser.add_argument(
        "--project",
        default="build/mininpu_v2_vitis",
        help="Vitis project directory (default: %(default)s)",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=100.0,
        help="Vitis target frequency in MHz (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="u280",
        help="Vitis target device (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.vitis:
        run_vitis(args.project, args.frequency, args.device)
    else:
        run_workload()


if __name__ == "__main__":
    main()
