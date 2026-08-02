"""Small int32 systolic GEMM used to exercise the ASIC flow."""

from pathlib import Path

import allo
import allo.dataflow as df
from allo.ir.types import int32, Stream


M = 4
N = 4
K = 4
P0 = M + 2
P1 = N + 2


@df.region()
def top(A: int32[M, K], B: int32[K, N], C: int32[M, N]):
    fifo_A: Stream[int32, 4][P0, P1]
    fifo_B: Stream[int32, 4][P0, P1]

    @df.kernel(mapping=[P0, P1], args=[A, B, C])
    def gemm(
        local_A: int32[M, K],
        local_B: int32[K, N],
        local_C: int32[M, N],
    ):
        i, j = df.get_pid()

        with allo.meta_if(i in {0, M + 1} and j in {0, N + 1}):
            pass
        with allo.meta_elif(j == 0):
            for k in range(K):
                fifo_A[i, j + 1].put(local_A[i - 1, k])
        with allo.meta_elif(i == 0):
            for k in range(K):
                fifo_B[i + 1, j].put(local_B[k, j - 1])
        with allo.meta_elif(i == M + 1 and j > 0):
            for k in range(K):
                fifo_B[i, j].get()
        with allo.meta_elif(j == N + 1 and i > 0):
            for k in range(K):
                fifo_A[i, j].get()
        with allo.meta_else():
            accumulator: int32 = 0
            for k in range(K):
                a: int32 = fifo_A[i, j].get()
                b: int32 = fifo_B[i, j].get()
                accumulator += a * b
                fifo_A[i, j + 1].put(a)
                fifo_B[i + 1, j].put(b)
            local_C[i - 1, j - 1] = accumulator


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
