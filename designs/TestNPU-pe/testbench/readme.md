PE testbench for the mflowgen VCS node.

`test_vectors.txt` is read with `$readmemh`. Each line is one 96-bit hex word:

```
{expected_fp32, psum_in_fp32, activation_bf16, weight_bf16}
```

BF16 operands use the upper 16 bits of their IEEE-754 FP32 representation,
matching the current MiniNPU `STORAGE_FORMAT="BF16"` configuration.
`generate_test_vectors.py` emits a small deterministic smoke set. You can replace
it with a cocotb/Python vector export as long as it writes the same line format
and keeps `NUM_TEST_VECTORS` in `design.args` in sync.
