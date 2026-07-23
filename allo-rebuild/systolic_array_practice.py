"""
Example systolic arrays to practice using allo

Julian Bushlow, 7/20/26
"""

import allo
import allo.dataflow as df
from allo.ir.types import int32, Stream
import numpy as np

NROWS, NCOLS, NHID = 4,4,4
DEPTH = 2

# Edition 1: explicit kernels to load inputs/receive results
# operation is gemm, A @ B, A: [NROWSxNHID], B: [NHIDxK]

@df.region()
def sys_array(A: int32[NROWS,NHID], B: int32[NHID,NCOLS], C: int32[NROWS,NCOLS]):

    # declare streams
    vert: Stream[int32,DEPTH][NROWS+1,NCOLS] # going down
    hori: Stream[int32,DEPTH][NROWS,NCOLS+1] # going right

    @df.kernel(mapping=[NROWS,NCOLS],args=[C])
    def pe(local_C: int32[NROWS,NCOLS]):
        i,j = df.get_pid()

        accum: int32 = 0
        for k in range(NHID):
            vert_incoming: int32 = vert[i,j].get()
            hori_incoming: int32 = hori[i,j].get()

            accum += vert_incoming * hori_incoming

            vert[i+1,j].put(vert_incoming)
            hori[i,j+1].put(hori_incoming)
        local_C[i, j] = accum

    # loaders
    
    @df.kernel(mapping=[NROWS],args=[A])
    def hori_loader(local_A: int32[NROWS,NHID]):
        i = df.get_pid()
        for k in range(NHID):
            hori[i,0].put(local_A[i,k])

    @df.kernel(mapping=[NCOLS],args=[B])
    def vert_loader(local_B: int32[NHID,NCOLS]):
        j = df.get_pid()
        for k in range(NHID):
            vert[0,j].put(local_B[k,j])

    # receive values - ensure no blocking

    @df.kernel(mapping=[1])
    def hori_receiver():
        with allo.meta_for(NROWS) as i:
            for k in range(NHID):
                unused_a: int32 = hori[i,NCOLS].get()

    @df.kernel(mapping=[1])
    def vert_receiver():
        with allo.meta_for(NCOLS) as j:
            for k in range(NHID):
                unused_b: int32 = vert[NROWS,j].get()

# Edition 2: Use meta_if/elif/else to do loading/returning
# operation is some random arithmetic
# Add PEs around edges which are used for loading/returning
# streams indexed as 
# computation PEs: get from "self", put to "next"
# loading PEs: just put to "self"
# assumes NROWS = NCOLS

@df.region()
def sys_array_gemm(A: int32[NROWS,NCOLS], B: int32[NROWS,NCOLS], C: int32[NROWS,NCOLS]):

    # declare streams
    vert: Stream[int32,DEPTH][NROWS+1,NCOLS] # going down
    hori: Stream[int32,DEPTH][NROWS,NCOLS+1] # going right

    @df.kernel(mapping=[NROWS+1,NCOLS+1],args=[A,B,C])
    def pe(local_A: int32[NROWS,NCOLS], local_B: int32[NROWS,NCOLS], local_C: int32[NROWS,NCOLS]):

        i, j = df.get_pid()
        
        # corner case: top left corner does nothing
        with allo.meta_if(i==0 and j==0):
            pass
        
        # load A from left side: 
        with allo.meta_elif(j==0):
            for k in range(NCOLS):
                hori[i-1,j].put(local_A[i-1,k])

        # load B from top: 
        with allo.meta_elif(i==0):
            for k in range(NROWS):
                vert[i,j-1].put(local_B[k,j-1])

        with allo.meta_else():
            accum: int32 = 0
            for k in range(NROWS): 
                a: int32 = hori[i-1,j-1].get()
                b: int32 = vert[i-1,j-1].get()
                accum += a * b

                # avoid forwarding if on edges
                with allo.meta_if(j < NCOLS):
                    hori[i - 1, j].put(a)
                with allo.meta_if(i < NROWS):
                    vert[i, j - 1].put(b)
            local_C[i-1,j-1] = accum 

# instantiate and test

rng = np.random.default_rng(seed=0)

# =========================================================
# Test 1: original meta_if-based square systolic array
# =========================================================

# This must match the dimension used when sys_array_gemm
# was defined.

A1 = rng.integers(
    low=-4,
    high=5,
    size=(NROWS, NCOLS),
    dtype=np.int32,
)
B1 = rng.integers(
    low=-4,
    high=5,
    size=(NROWS, NCOLS),
    dtype=np.int32,
)
C1 = np.zeros(
    (NROWS, NCOLS),
    dtype=np.int32,
)

expected1 = A1 @ B1

print("=" * 60)
print("TEST 1: meta_if-based square systolic array")
print("=" * 60)

sim1 = df.build(sys_array_gemm, target="simulator")
sim1(A1, B1, C1)

print("A:")
print(A1)

print("\nB:")
print(B1)

print("\nSystolic result:")
print(C1)

print("\nNumPy reference:")
print(expected1)

np.testing.assert_array_equal(C1, expected1)
print("\nTEST 1 PASSED")

# =========================================================
# Test 2: explicit loader/drain systolic array
# =========================================================

A2 = rng.integers(
    low=-4,
    high=5,
    size=(NROWS, NHID),
    dtype=np.int32,
)
B2 = rng.integers(
    low=-4,
    high=5,
    size=(NHID, NCOLS),
    dtype=np.int32,
)
C2 = np.zeros(
    (NROWS, NCOLS),
    dtype=np.int32,
)

expected2 = A2 @ B2

print("\n")
print("=" * 60)
print("TEST 2: explicit loader/drain systolic array")
print("=" * 60)

sim2 = df.build(sys_array, target="simulator")
sim2(C2, A2, B2)

print("A:")
print(A2)

print("\nB:")
print(B2)

print("\nSystolic result:")
print(C2)

print("\nNumPy reference:")
print(expected2)

np.testing.assert_array_equal(C2, expected2)
print("\nTEST 2 PASSED")

print("\n")
print("=" * 60)
print("ALL SYSTOLIC ARRAY TESTS PASSED")
print("=" * 60)
