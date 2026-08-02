"""Small int32 streaming GEMM used to exercise the ASIC flow."""

from pathlib import Path

import allo
import allo.dataflow as df
import numpy as np
from allo.ir.types import int32, Stream


M = 4
N = 4
K = 4
DEPTH = 4


@df.region()
def top(A: int32[M * K], B: int32[K * N], C: int32[M * N]):
    """Compute C = A @ B with a 2-D systolic array.

    The external arrays are deliberately one-dimensional: this is a common HLS
    memory interface, and it lets nested dataflow kernels use the same type with
    no generated whole-array I/O wrapper.  The loaders and result writer are the
    only processes that access external memory; compute PEs communicate solely
    through point-to-point streams.
    """

    horizontal: Stream[int32, DEPTH][M, N + 1]
    vertical: Stream[int32, DEPTH][M + 1, N]
    result: Stream[int32, DEPTH][M, N]

    @df.kernel(mapping=[1], args=[A])
    def load_a(local_A: int32[M * K]):
        with allo.meta_for(M) as i:
            for k in range(K):
                horizontal[i, 0].put(local_A[i * K + k])

    @df.kernel(mapping=[1], args=[B])
    def load_b(local_B: int32[K * N]):
        with allo.meta_for(N) as j:
            for k in range(K):
                vertical[0, j].put(local_B[k * N + j])

    @df.kernel(mapping=[M, N])
    def compute():
        i, j = df.get_pid()
        accumulator: int32 = 0
        for k in range(K):
            a: int32 = horizontal[i, j].get()
            b: int32 = vertical[i, j].get()
            accumulator += a * b
            with allo.meta_if(j < N - 1):
                horizontal[i, j + 1].put(a)
            with allo.meta_if(i < M - 1):
                vertical[i + 1, j].put(b)
        result[i, j].put(accumulator)

    @df.kernel(mapping=[1], args=[C])
    def store_c(local_C: int32[M * N]):
        with allo.meta_for(M) as i:
            with allo.meta_for(N) as j:
                local_C[i * N + j] = result[i, j].get()


def build(project, target, mode, configs):
    """Build using the standard allo-asic-compilation node contract."""
    project = Path(project).resolve()
    module = df.build(
        top,
        target=target,
        mode=mode,
        project=project,
        wrap_io=False,
        configs=configs,
    )
    module()


def workload():
    """Return one deterministic transaction and its expected result.

    Testbench generation can serialize this backend-independent description
    after HLS determines the concrete RTL interface protocol and port names.
    Array values use the same flattened layout as the hardware top function.
    """
    matrix_a = np.array(
        [
            [1, 2, 3, 4],
            [-1, 0, 2, 1],
            [3, -2, 1, 0],
            [2, 1, -1, 3],
        ],
        dtype=np.int32,
    )
    matrix_b = np.array(
        [
            [2, 0, 1, -1],
            [1, 3, -2, 2],
            [0, 1, 4, 1],
            [-2, 2, 0, 3],
        ],
        dtype=np.int32,
    )
    expected_c = matrix_a @ matrix_b
    return {
        "inputs": {
            "A": matrix_a.reshape(-1),
            "B": matrix_b.reshape(-1),
        },
        "expected_outputs": {"C": expected_c.reshape(-1)},
    }


def run_workload():
    """Run and check the workload with Allo's dataflow simulator."""
    transaction = workload()
    a = transaction["inputs"]["A"].copy()
    b = transaction["inputs"]["B"].copy()
    c = np.zeros(M * N, dtype=np.int32)

    simulator = df.build(top, target="simulator")
    simulator(a, b, c)
    np.testing.assert_array_equal(c, transaction["expected_outputs"]["C"])
    print("Allo streaming GEMM workload passed")


if __name__ == "__main__":
    run_workload()
