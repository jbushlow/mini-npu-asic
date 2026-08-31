import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).with_name("generate_test_vectors.py")
    spec = importlib.util.spec_from_file_location("generate_test_vectors", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vector_word_encodes_bf16_operands_and_fp32_result():
    module = load_module()
    assert module.vector_word(1.0, 1.0, 0.0) == "3f800000000000003f803f80"
