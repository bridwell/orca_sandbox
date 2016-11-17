import pytest
from ..collector import *


def test_collect_inputs_noArgs():
    """
    Test the collect inputs method
    for the simple case where the target
    function does not have any parameters.

    """
    # define a function to test with
    def my_func1():
        return 1

    # 1st test a function with no inputs provided
    res_kwargs = collect_inputs(my_func1)
    assert len(res_kwargs) == 0
    assert my_func1(**res_kwargs) == 1

    # next test when the calling function passes args
    # not defined by the called function, these
    # should be ignored
    kwargs = {'some_val': 50}
    res_kwargs = collect_inputs(my_func1, **kwargs)
    assert len(res_kwargs) == 0
    assert my_func1(**res_kwargs) == 1


def test_collect_inputs_withArgs():
    """
    Test the collect inputs method where a function
    has input arguments. These arguments may be provided
    from the calling function, a global dictionary
    of variables or the function defaults.

    """
    # function to test with
    def times(val1, val2=10):
        return val1 * val2

    #  global injectables available
    global_injs = {
        'val1': 200,
        'val2': 500
    }

    # define some local kwargs from a hypothetical call
    kwargs = {'val1': 5}

    # 1st test w/ local kwargs, these should override
    # the globals, the globals should override
    # the function default
    res_kwargs = collect_inputs(times, global_injs, **kwargs)
    assert len(res_kwargs) == 2
    assert res_kwargs['val1'] == kwargs['val1']
    assert res_kwargs['val2'] == global_injs['val2']
    assert times(**res_kwargs) == kwargs['val1'] * global_injs['val2']

    # next test w/ kwargs absent, should use globals for all
    res_kwargs = collect_inputs(times, global_injs)
    assert len(res_kwargs) == 2
    assert res_kwargs['val1'] == global_injs['val1']
    assert res_kwargs['val2'] == global_injs['val2']
    assert times(**res_kwargs) == global_injs['val1'] * global_injs['val2']

    # test using no globals, uses local and defaults
    res_kwargs = collect_inputs(times, **kwargs)
    assert len(res_kwargs) == 2
    assert res_kwargs['val1'] == kwargs['val1']
    assert res_kwargs['val2'] == 10
    assert times(**res_kwargs) == kwargs['val1'] * 10

    # make sure an error is raised when args not found
    with pytest.raises(ValueError):
        collect_inputs(times)


def test_collect_inputs_within_class():
    """
    Make sure we can collect inputs when
    the calling function is within a class.

    """

    class my_plus():

        def __call__(self, val1, val2):
            return val1 + val2

    mc = my_plus()
    res_kwargs = collect_inputs(
        mc.__call__,
        {'val1': 20},
        **{'val2': 50}
    )

    assert len(res_kwargs) == 2
    assert res_kwargs['val1'] == 20
    assert res_kwargs['val2'] == 50
    assert mc(**res_kwargs) == 70


def test_collectable():
    """
    Test collection of global injectable
    object that implements Collectable.

    """

    # define a collectable class
    class my_collect1(Collectable):

        def __init__(self):
            self.value = 10

        def __call__(self):
            return self.value

    # define another class that overrides
    class my_collect2(Collectable):

        def __init__(self):
            self.value1 = 5
            self.value2 = 20

        def __call__(self):
            return self.value1

        def collect(self):
            return self.value2

    # define function that has collectables as args
    def my_func(obj1, obj2):
        return obj1 + obj2

    # inject class instances
    global_injs = {
        'obj1': my_collect1(),
        'obj2': my_collect2()
    }

    # do the collection and evaluation
    res_kwargs = collect_inputs(my_func, global_injs)
    print res_kwargs
    assert len(res_kwargs) == 2
    assert res_kwargs['obj1'] == 10
    assert res_kwargs['obj2'] == 20
    assert my_func(**res_kwargs) == 30
