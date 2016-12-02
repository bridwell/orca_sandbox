import pytest
import numpy as np
import pandas as pd

from .. import orca


@pytest.fixture()
def a_value():
    return 100


@pytest.fixture()
def my_func():
    def my_func(value_test1):
        return value_test1 * 2
    return my_func


@pytest.fixture()
def my_func2():
    def my_func2(func_test1):
        return func_test1 + func_test1
    return my_func2


@pytest.fixture()
def my_class():
    class my_class(object):

        def __init__(self, value_test1):
            self.val = value_test1

        def __call__(self, func_test1):
            return self.val + func_test1
    return my_class


def test_injectables(a_value, my_func, my_func2, my_class):
    """
    Test wrapping values and functions,
    also evaluating them and propogating
    changes.

    """

    # clear everything
    orca.clear_all()

    # inject a value
    orca.add_injectable('value_test1', a_value)

    # inject a function with a value dependency
    orca.add_injectable(
        'func_test1',
        my_func,
        clear_on='value_test1'
    )

    # inject a function with a function dependency
    orca.add_injectable(
        'func_test2',
        my_func2,
        clear_on='func_test1'
    )

    # inject a callable class with dependencies
    orca.add_injectable(
        'class_test',
        my_class
    )

    # check the registration
    injs = ['value_test1', 'func_test1', 'func_test2', 'class_test']
    for inj in orca.list_injectables():
        assert inj in injs

    # evaluate functions
    assert orca.eval_injectable('value_test1') == 100
    assert orca.eval_injectable('func_test1') == 200
    assert orca.eval_injectable('func_test2') == 400
    assert orca.eval_injectable('class_test') == 300

    # update the value injectable and check again
    orca.add_injectable('value_test1', 5)
    assert orca.eval_injectable('value_test1') == 5
    assert orca.eval_injectable('func_test1') == 10
    assert orca.eval_injectable('func_test2') == 20

    """
    Note that this retains the original value_test1 value (100).
    Only __call__ arguments will be re-evaluated. Arguments
    passed in for initialization will only be evaluated once.
    """
    assert orca.eval_injectable('class_test') == 110


def test_callbacks(my_func2):
    """
    For injectables where autocall is False.

    """
    orca.clear_all()
    orca.add_injectable('cb_test', my_func2, autocall=False)
    cb = orca.eval_injectable('cb_test')
    assert cb(10) == 20


def test_columns():
    """
    Test creating columns.

    """
    orca.clear_all()

    def get_s(cnt):
        return pd.Series(np.arange(cnt))

    # wrap a series
    s_4 = get_s(4)
    orca.add_column('series1', s_4, attach_to='something')

    # wrap a function that returns a series
    orca.add_injectable('cnt', 2)
    orca.add_column('series2', get_s, clear_on='cnt',
                    attach_to=['something', 'something_else'])

    # check evaluations
    res1 = orca.eval_injectable('series1')
    assert (res1 == get_s(4)).all()

    res2 = orca.eval_injectable('series2')
    assert (res2 == get_s(2)).all()

    orca.add_injectable('cnt', 5)
    res3 = orca.eval_injectable('series2')
    assert (res3 == get_s(5)).all()

    # check registered attachements
    a = orca._attachments
    assert len(a) == 2
    assert len(a['something']) == 2
    assert len(a['something_else']) == 1
