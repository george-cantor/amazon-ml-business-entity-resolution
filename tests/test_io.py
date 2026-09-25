from src.io import read_tsv


def test_reader_function_exists():
    assert callable(read_tsv)