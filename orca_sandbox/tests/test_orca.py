import os
import tempfile
import pytest
import numpy as np
import pandas as pd

from .. import orca


def test_get_args():
    """
    Tests extracting argument names and defaults
    from a function signature.

    """
    def f(a, b, c=10):
        return

    args, defaults = orca._get_func_args(f)
    assert args == ['a', 'b', 'c']
    assert defaults == {'c': 10}
