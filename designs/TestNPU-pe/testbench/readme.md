PE testbench for the mflowgen VCS node.

`test_vectors.txt` is read with `$readmemh`. Each line is one 96-bit hex word:

```
{expected_fp32, psum_in_fp32, activation_fp16, weight_fp16}
```

`generate_test_vectors.py` emits a small deterministic smoke set. You can replace
it with a cocotb/Python vector export as long as it writes the same line format
and keeps `NUM_TEST_VECTORS` in `design.args` in sync.
