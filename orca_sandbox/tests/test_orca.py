import pytest

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
    for inj in injs:
        assert inj in injs

    # evaluate functions
    assert orca.eval_injectable('value_test1') == 100
    assert orca.eval_injectable('func_test1') == 200
    assert orca.eval_injectable('func_test2') == 400
    assert orca.eval_injectable('class_test') == 300

    # update the value injectable and check again
    orca.update_injectable('value_test1', 5)
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

    Note right, now get_injectable, returns the
    injectable itself and not the function. That
    maybe should change??

    The notion of autocall and callback should probably be
    refactored into its own class?

    """
    orca.clear_all()
    orca.add_injectable('cb_test', my_func2, autocall=False)
    cb = orca.get_injectable('cb_test')
    assert cb(func_test1=10) == 20
