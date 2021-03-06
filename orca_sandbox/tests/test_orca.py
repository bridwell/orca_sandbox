import os
import tempfile
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


def test_register_clear_events(my_func2):
    orca.clear_all()

    # test w/ no clear events
    orca._register_clear_events(my_func2, None)
    assert my_func2 in orca._events['env'].subscribers

    # test with a single event
    orca._register_clear_events(my_func2, 'test1')
    assert my_func2 in orca._events['test1'].subscribers

    # test with an attachment
    orca._register_clear_events(my_func2, 'table.column')
    assert my_func2 in orca._events['table'].subscribers
    assert my_func2 in orca._events['table.column'].subscribers


def test_get_args():

    def f(a, b, c=10):
        return

    args, defaults = orca._get_func_args(f)
    assert args == ['a', 'b', 'c']
    assert defaults == {'c': 10}


def test_collect_inputs():

    orca.clear_all()
    orca.add_injectable('a', 100)
    orca.add_injectable('b', 200)
    orca.add_injectable('one', 1)
    orca.add_injectable('two', 2)

    def a_plus_b(a, b):
        return a + b

    # test 1 - a function with arguments that match injected
    kwargs = orca._collect_inputs(a_plus_b)
    assert kwargs == {'a': 100, 'b': 200}

    # test 2 - with argmap
    arg_map = {'a': 'one', 'b': 'two'}
    kwargs = orca._collect_inputs(a_plus_b, arg_map)
    assert kwargs == {'a': 1, 'b': 2}

    # test 3 - with local kwargs
    kwargs = orca._collect_inputs(a_plus_b, **{'a': 10, 'b': 20})
    assert kwargs == {'a': 10, 'b': 20}

    # test 4 - arguments not found
    orca.clear_all()
    with pytest.raises(ValueError):
        orca._collect_inputs(a_plus_b)


def test_get_callable(my_func2, my_class):
    orca.clear_all()

    # function
    f = orca._get_callable(my_func2)
    assert f.__name__ == my_func2.__name__

    # class
    orca.add_injectable('value_test1', 10)
    f = orca._get_callable(my_class)
    assert f.__name__ == '__call__'

    # a non-function
    with pytest.raises(ValueError):
        orca._get_callable(10)


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
    assert orca.eval_injectable('func_test2') == 20
    assert orca.eval_injectable('func_test1') == 10

    """
    Note that this retains the original value_test1 value (100).
    Only __call__ arguments will be re-evaluated. Arguments
    passed in for initialization will only be evaluated once.
    """
    assert orca.eval_injectable('class_test') == 110

    # check clearing the environment
    orca.clear_all()
    print orca.list_injectables()
    assert len(orca.list_injectables()) == 0
    assert len(orca._attachments) == 0
    for k, v in orca._events.items():
        assert len(v.subscribers) == 0


def test_upstream_injectables():
    """
    Testing the propagation of changes for upstream cases.

    """
    orca.clear_all()

    @orca.injectable(clear_on='a')
    def b(a):
        return a + 1

    @orca.injectable('c', clear_on='b')
    def func_c(b):
        return b * -1

    # test 1 - upstream is a value
    orca.add_injectable('a', 10)
    assert orca.eval_injectable('c') == -11
    orca.add_injectable('a', 20)
    assert orca.eval_injectable('c') == -21

    # test 2 - upstream is a function
    temp = [0]

    @orca.injectable()
    def a():
        temp[0] += 1
        return temp[0]

    assert orca.eval_injectable('c') == -2
    assert orca.eval_injectable('c') == -3


def test_callbacks(my_func2):
    """
    For injectables where autocall is False.

    """
    orca.clear_all()

    @orca.callback()
    def cb_test(arg1):
        return arg1 + arg1

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
    @orca.column(attach_to='a_df')
    def twos():
        return pd.Series(np.ones(row_count)) * 2

    # wrap a function that returns a data frame
    @orca.table('blah_df', attach_to='a_df')
    def get_blah():
        return pd.DataFrame({
            'threes': np.ones(row_count) * 3,
            'fours': np.ones(row_count) * 4,
        })

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

    # list the tables
    the_tables = orca.list_tables()
    assert len(the_tables) == 2
    assert 'a_df' in the_tables
    assert 'blah_df' in the_tables

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


def test_update_table_columns():
    """
    Make sure changes to table values
    are propogated.

    """
    orca.clear_all()

    # wrap a table we will be updating
    df = pd.DataFrame({
        'a': np.zeros(4),
        'b': np.ones(4)
    })
    orca.add_table('df1', df)

    # wrap a column function that depends on column 'b'
    @orca.column('test_func', clear_on='df1.b')
    def my_func(df1):
        return df1['b'] * 2.0

    # initial evaluation
    assert (orca.eval_injectable('test_func') == 2).all()

    # update the entire column and re-evaluate
    tab = orca.eval_injectable('df1')
    tab['b'] = 5
    assert (orca.eval_injectable('test_func') == 10).all()

    # partial column update using 'update_col_from_series'
    tab = orca.eval_injectable('df1')
    to_update = pd.Series([20, 20], index=pd.Index([1, 3]))
    tab.update_col_from_series('b', to_update)
    expected = [10, 40, 10, 40]
    assert (orca.eval_injectable('test_func') == expected).all()

    # update the entire table
    orca.add_injectable('df1', pd.concat([tab.local, tab.local]))
    assert len(orca.eval_injectable('test_func')) == 8


def test_run_simple():
    """
    Simple test of step execution, doesn't write out anything.

    """

    orca.add_injectable('test1', 10)
    orca.add_injectable('test2', 20)

    @orca.step()
    def my_step1(test1):
        orca.add_injectable('test1', test1 + 10)

    @orca.step('my_step2')
    def my_step2_func(test2):
        orca.add_injectable('test2', test2 + 1)

    steps = orca.list_steps()
    assert len(steps) == 2
    assert 'my_step1' in steps
    assert 'my_step2' in steps

    orca.run(
        steps=[
            'my_step1',
            ('my_step2', [2010, 2011])
        ],
        iter_vars=[2010, 2011, 2012]
    )

    assert orca.eval_injectable('test1') == 40
    assert orca.eval_injectable('test2') == 22


@pytest.fixture
def store_name(request):
    fname = tempfile.NamedTemporaryFile(suffix='.h5').name

    def fin():
        if os.path.isfile(fname):
            os.remove(fname)
    request.addfinalizer(fin)

    return fname


def test_write_load_tables(store_name):
    orca.clear_all()

    #######################
    # TEST WRITING TABLES
    #######################

    # add some tables
    orca.add_table(
        'tab1',
        pd.DataFrame({
            'a': np.ones(4),
            'b': np.zeros(4)
        })
    )
    orca.add_column(
        'col1',
        pd.Series(
            np.random.randint(0, 10, 4)
        ),
        attach_to='tab1'
    )

    orca.add_table(
        'tab2',
        pd.DataFrame({
            'c': np.random.rand(5),
            'd': np.random.rand(5)
        })
    )

    # write out tables a couple different ways
    orca.write_tables(store_name, write_attached=True)
    orca.write_tables(store_name, ['tab1'], 2010)
    orca.write_tables(store_name, ['tab2'], 2020)

    # check the store
    with pd.get_store(store_name, mode='r') as store:
        assert 'tab1' in store
        assert 'tab2' in store
        assert '2010/tab1' in store
        assert '2020/tab2' in store

        assert 'col1' in store['tab1'].columns
        assert 'col1' not in store['2010/tab1'].columns

    #######################
    # TEST LOADING TABLES
    #######################

    def assert_load_tables(expected_tabs, *args, **kwargs):
        """
        Compares expected tables with the tables that
        get registered via differet arg inputs.
        """
        orca.clear_all()
        orca.load_tables(*args, **kwargs)
        o_tabs = orca.list_tables()

        assert len(o_tabs) == len(expected_tabs)
        for t in expected_tabs:
            assert t in o_tabs

    # test 1- ways to load the whole store
    e_tabs = ['/tab1', '/tab2', '/2010/tab1', '/2020/tab2']
    assert_load_tables(e_tabs, store_name)
    assert_load_tables(e_tabs, store_name, '/*')

    # test 2 - apply pre-fix filters different ways
    assert_load_tables(['/2010/tab1'], store_name, '2010')
    assert_load_tables(['/2010/tab1'], store_name, '/2010')
    assert_load_tables(['tab1'], store_name, '2010/', remove_prefix=True)
    assert_load_tables(['/tab1', '/tab2'], store_name, '/')
    assert_load_tables([], store_name, '/2030*')

    # test 3 - query by base table name
    assert_load_tables(['/tab1', '/2010/tab1'], store_name, basenames=['tab1'])

    # test 4 - query by prefix and basetables
    assert_load_tables(['tab1'], store_name, prefix='/', basenames=['tab1'], remove_prefix=True)


def test_run_and_write(store_name):
    orca.clear_all()

    @orca.table()
    def a_table():
        return pd.DataFrame({'col1': np.zeros(1)})

    @orca.table()
    def b_table():
        return pd.DataFrame({'my_col': ['a', 'b', 'c']})

    @orca.table()
    def c_table():
        return pd.DataFrame({'r': np.random.rand(4)})

    @orca.step()
    def plus1(a_table):
        orca.add_table(
            'a_table', a_table.to_frame() + 1)

    tabs = orca.list_tables()

    orca.run(
        ['plus1'],
        range(2015, 2021),
        data_out=store_name,
        out_interval=2,
        out_base_tables=['a_table', 'b_table', 'c_table'],
        out_run_tables=['a_table']
    )

    # check the base tables output
    orca.clear_all()
    orca.load_tables(store_name, prefix='base', remove_prefix=True)
    o_tabs = orca.list_tables()
    assert len(o_tabs) == 3
    for t in tabs:
        assert t in o_tabs

    # check the run tables output
    orca.clear_all()
    orca.load_tables(store_name, basenames=['a_table'])
    o_tabs = orca.list_tables()
    assert len(o_tabs) == 5

    def check_sums(y, total):
        name = '/{}/a_table'.format(y)
        df = orca.eval_injectable(name).local
        assert df['col1'].sum() == total

    check_sums('base', 0)
    check_sums(2015, 1)
    check_sums(2017, 3)
    check_sums(2019, 5)
    check_sums(2020, 6)
