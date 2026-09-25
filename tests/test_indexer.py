from codeqa import config
from codeqa.indexer import python_chunks, window_chunks


def test_functions_and_module_code_are_separate_chunks():
    src = "import os\nX = 1\n\n@deco\ndef f():\n    return 1\n\nclass C:\n    pass\n"
    chunks = list(python_chunks("a.py", src))
    by_symbol = {c.symbol: (c.start_line, c.end_line) for c in chunks}
    assert by_symbol["f"] == (4, 6)          # decorator included
    assert by_symbol["C"] == (8, 9)
    assert by_symbol["<module>"] == (1, 3)


def test_large_class_is_split_per_method():
    methods = "".join(f"    def m{i}(self):\n" + "        x = 1\n" * 30 for i in range(6))
    src = "class Big:\n    '''doc'''\n" + methods
    symbols = [c.symbol for c in python_chunks("b.py", src)]
    assert symbols[0] == "Big"
    assert symbols[1:] == [f"Big.m{i}" for i in range(6)]


def test_syntax_error_falls_back_to_windows():
    chunks = list(python_chunks("c.py", "def broken(:\n  pass\n"))
    assert [c.symbol for c in chunks] == ["<window>"]


def test_windows_overlap_and_cover_everything():
    lines = [f"line {i}" for i in range(1, 131)]
    chunks = list(window_chunks("d.txt", lines, 1, 130))
    assert chunks[0].start_line == 1 and chunks[-1].end_line == 130
    step = config.WINDOW_LINES - config.WINDOW_OVERLAP
    assert chunks[1].start_line == 1 + step
