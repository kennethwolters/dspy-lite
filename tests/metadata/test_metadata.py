import re

import dspy


def test_metadata():
    assert dspy.__name__ == "dspy"
    assert dspy.__package_name__ == "dspy-lite"
    assert re.match(r"\d+\.\d+\.\d+", dspy.__version__)
    assert dspy.__author__ == "Kenneth Wolters"
    assert dspy.__url__ == "https://github.com/kennethwolters/dspy-lite"
