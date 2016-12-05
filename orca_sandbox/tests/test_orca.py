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


def test_column_wrapper():
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


def test_table_wrappper():
    orca.clear_all()
    row_count = 5

    # wrap and attach a column/series
    orca.add_column('ones', pd.Series(np.ones(row_count)), attach_to='a_df')

    # wrap a function that returns a series and attach it
    def twos():
        return pd.Series(np.ones(row_count)) * 2
    orca.add_column('twos', twos, attach_to='a_df')

    # wrap a function that returns a data frame
    df = pd.DataFrame({
        'threes': np.ones(row_count) * 3,
        'fours': np.ones(row_count) * 4,
    })
    orca.add_table('blah_df', df, attach_to='a_df')

    # evaluate the stuff we've wrapped
    assert (orca.eval_injectable('ones') == 1).all()
    assert (orca.eval_injectable('twos') == 2).all()

    blah = orca.eval_injectable('blah_df')
    assert len(blah) == row_count
    assert 'threes' in blah.local_columns
    assert 'fours' in blah.local_columns
    assert blah.attached_columns == []
    assert (blah.index.values == np.arange(row_count)).all()
    assert (blah.threes == 3).all()
    assert (blah['fours'] == 4).all()

    # wrap a target data frame that we are attaching stuff to
    target = pd.DataFrame({
        'fives': np.ones(row_count) * 5,
        'sixes': np.ones(row_count) * 6,
    })
    orca.add_table('a_df', target)

    # evaluate the target table
    tab = orca.eval_injectable('a_df')
    expected = {
        'ones': 1,
        'twos': 2,
        'threes': 3,
        'fours': 4,
        'fives': 5,
        'sixes': 6
    }

    for c in expected.keys():
        assert c in tab.columns
    for c in ['fives', 'sixes']:
        assert c in tab.local_columns
    for c in ['ones', 'twos', 'threes', 'fours']:
        assert c in tab.attached_columns

    # check grabbing all attached columns
    df_all = tab.to_frame()
    for c in expected.keys():
        assert c in df_all.columns
    for k, v in expected.items():
        assert (df_all[k] == v).all()

    # check grabbing just local columns
    c = ['fives', 'sixes']
    df_local = tab.to_frame(c)
    assert (df_local.columns == c).all()
    assert (df_local.fives == 5).all()
    assert (df_local['sixes'] == 6).all()

    # check grabbing just attached
    c = ['ones', 'threes']
    df_a = tab.to_frame(c)
    assert (df_a.columns == c).all()
    assert (df_a.ones == 1).all()
    assert (df_a.threes == 3).all()

    # check grabbing both local and attached
    c = ['fours', 'twos', 'sixes']
    df_a = tab.to_frame(c)
    assert (df_a.columns == c).all()
    assert (df_a.fours == 4).all()
    assert (df_a.twos == 2).all()
    assert (df_a.sixes == 6).all()

    # test missing columns
    with pytest.raises(ValueError):
        tab.to_frame(['zeros', 'ones', 'twos'])
