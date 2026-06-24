import re
from pathlib import Path

import dspy


def test_metadata():
    assert dspy.__name__ == "dspy"
    assert dspy.__package_name__ == "dspy-lite"
    assert re.match(r"\d+\.\d+\.\d+", dspy.__version__)
    assert dspy.__version__ == "3.2.1.post1"
    assert dspy.__author__ == "Kenneth Wolters"
    assert dspy.__url__ == "https://github.com/kennethwolters/dspy-lite"
    assert dspy.__description__ == "DSPy with litellm replaced by litelm and numpy made optional"


def test_project_metadata_matches_runtime_metadata():
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    text = pyproject.read_text()

    assert 'name = "dspy-lite"' in text
    assert f'version = "{dspy.__version__}"' in text
    assert f'description = "{dspy.__description__}"' in text
    assert f'authors = [{{ name = "{dspy.__author__}" }}]' in text
    assert f'Homepage = "{dspy.__url__}"' in text
