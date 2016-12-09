"""
Trying out an over-hauled verion of orca, as a potential
simulation orchestration framework.

This is very experimental.

"""
import os
import time
import inspect
import pandas as pd

from .events import *
# from .collector import *


########################
# PRIVATE MODULE-LEVEL
########################

def _init_globals():
    """
    Initializes module-level variables.

    """
    global _injectables, _events, _attachments, _attached
    _injectables = {}
    _attachments = {}  # keyed by the tables attached to
    _attached = {}     # keyed by the tables attached from
    _events = init_events(['env', 'run', 'iteration', 'step', 'collect'])


# initialize the globals upon importing
# is there a better way to do this?
_init_globals()


def _notify_changed(name):
    """
    Notifies all subscribers of a state change.

    Parameters:
    -----------
    name: str
        The name of the event to fire.

    """
    print '{} was fired!'.format(name)
    if name in _events:
        _events[name]()


def _register_clear_events(func, clear_on):
    """
    Registers cache clearing events. A 'env'
    event will always be registered as well.

    Parameters:
    -----------
    func: function
        Function to register. The function should clear
        cached data.
    clear_on: str or list of str
        Events that will trigger the func.

    """
    if clear_on is None:
        clear_on = ['env']
    if isinstance(clear_on, str):
        clear_on = [clear_on]
    if 'env' not in clear_on:
        clear_on = ['env'] + clear_on

    # if the the clear on has ., also register the left hand side
    # intended so that injectables depending on columns also depend
    # on the table
    more_clears = set()
    for c in clear_on:
        if '.' in c:
            more_clears.add(c.split('.')[0])
    clear_on += list(more_clears)

    subscribe_to_events(_events, clear_on, func, _collect_inputs)


def _get_func_args(func):
    """
    Returns a function's argument names and defaults.

    Parameters:
    -----------
    func: callable
        The function/callable to inspect.

    Returns:
    --------
    arg_names: list of str
        List of argument names.
    default_kwargs:
        Dictionary of default values. Keyed by the argument name.
    """

    # get function arguments
    spec = inspect.getargspec(func)
    args = spec.args

    # get defaults
    defaults = spec.defaults

    # get keyword args for the function's default values
    default_kwargs = {}
    if defaults is not None:
        kw_start_idx = len(args) - len(defaults)
        default_kwargs = dict(zip([key for key in args[kw_start_idx:]], list(defaults)))

    return args, default_kwargs


def _collect_inputs(func, arg_map={}, **local_kwargs):
    """
    Collect the inputs needed to execute a function.

    ** STILL NEED TO FIGURE OUT WAHT TO DO WITH DEFAULTS???

    Parameters:
    -----------
    func: callable:
        The function callable to execute.
    arg_map: dict, optional, default {}
        Dictionary that maps between the argument
        names of the function (keys) and the corresponding
        injectables (values).
        For example:
            def my_func(a):
                ...
            arg_map = {'a': 'my_injectable'}
        Would substitute argument 'a' with the value provided by
        'my_injectable.'
    **kwargs:
        Optional keyword arguments to provide to the function.
        These will overide any injected values.

    Returns:
    --------
    Named keyword arg dictionary that can be passed to execute the function.

    """
    kwargs = {}

    print arg_map

    # get function signature
    arg_names, defaults = _get_func_args(func)

    # if no args are needed by the function then we're done
    if len(arg_names) == 0:
        return kwargs

    # loop the through the function args and find the matching input
    for a in arg_names:

        if a == 'self':
            # call coming from a class, ignore
            continue

        # local inputs (i.e. keywords from the calling function)
        if a in local_kwargs:
            kwargs[a] = local_kwargs[a]
            continue

        # fetch from injectables
        name = a
        if a in arg_map:
            name = arg_map[a]

        if name in _injectables:
            inj = _injectables[name]

            if callable(inj):
                inj = inj()

            kwargs[a] = inj
            continue

        # argument not found
        raise ValueError("Argument {} not found".format(a))

    return kwargs


def _get_callable(wrapped):
    """
    Given a callable (function or class with __call__ method defined),
    returns the call function.

    Note: if the callable is a class, this assumes objects are created
    via the __init__ method. TODO: add checks for when this is absent
    and/or a __new__ method is provided instead.

    Parameters
    ----------
    wrapped: function or callable class

    Returns
    -------
    function

    """

    if inspect.isfunction(wrapped):
        # a function is wrapped
        return wrapped
    elif callable(wrapped):
        # a class is wrapped, need to create an instance first
        init_kwargs = _collect_inputs(wrapped.__init__)
        obj = wrapped(**init_kwargs)
        return obj.__call__
    else:
        raise ValueError('The wrapped argument must be a function or callable class.')


def _create_injectable(name, wrapped, clear_on=None, autocall=True, arg_map={}):
    """
    Creates an injectable from a provided value or function.

    """

    if not callable(wrapped):
        return ValueWrapper(name, wrapped)
    if not autocall:
        return CallbackWrapper(name, wrapped)
    if clear_on:
        return FuncWrapper(name, wrapped, clear_on, arg_map)
    else:
        return FuncWrapper(name, wrapped, 'collect', arg_map)


def _attach(name, attach_to):
    """
    Links an injectable with other injectables it it attached to. Used to
    bind columns or tables with other tables.

    """

    if attach_to is None:
        return

    def attach(target_name):
        # 1st attach to
        if target_name in _attachments:
            a = _attachments[target_name]
        else:
            a = set()
            _attachments[target_name] = a
        a.add(name)

        # next attach from
        if name in _attached:
            print 'attaching to {}'.format(name)
            a2 = _attached[name]
        else:
            a2 = set()
            _attached[name] = a2
        a2.add(target_name)

    if isinstance(attach_to, list):
        for target_name in attach_to:
            attach(target_name)
    else:
        attach(attach_to)


########################
# WRAPPER CLASSES
########################


class ValueWrapper(object):
    """
    Wraps a value.

    """

    def __init__(self, name, wrapped):
        self.name = name
        self._data = wrapped

    def __call__(self):
        return self._data


class CallbackWrapper(object):
    """
    Wraps a callback function...Still trying to work out what I want
    this do.

    Todo: add  memoize? Also can we inherit from FuncWrapper and just
    overide __call__?

    Right, now this notifies of changes each time it is collected.
    Not sure if that is the desired behavior. This would likely be used
    mostly in steps, so maybe that doesn't matter anyway.

    Should we just move this into the funcwrapper?

    """

    def __init__(self, name, wrapped):
        self.name = name
        self.func = _get_callable(wrapped)

    def __call__(self, **local_kwargs):
        f = self.func
        _notify_changed(self.name)
        return f


class FuncWrapper(object):
    """
    Wraps a function that supports caching.

    """

    def __init__(self, name, wrapped, clear_on, arg_map={}):
        self.name = name
        self.func = _get_callable(wrapped)
        self.arg_map = arg_map

        # set up caching
        self._data = None
        _register_clear_events(self.clear_cache, clear_on)

    def __call__(self, **local_kwargs):

        # clear out non-cached functions
        _notify_changed('collect')

        # if kwargs are provided, do a temporary evaluation
        # any cached data will be reverted to on the next call
        if len(local_kwargs) > 0:
            kwargs = _collect_inputs(self.func, self.arg_map, **local_kwargs)
            return self.func(**kwargs)

        # evaluate the function if not cached
        if self._data is None:
            kwargs = _collect_inputs(self.func, self.arg_map)
            self._data = self.func(**kwargs)

        return self._data

    def clear_cache(self):
        if self._data is not None:
            self._data = None
            _notify_changed(self.name)

            # notify things this is attached to?
            if self.name in _attached:
                for a in _attached[self.name]:
                    _notify_changed('{}.*'.format(a))


class StepFuncWrapper(object):
    """
    Wraps a callable with side-effects...

    Are there additional things we need here?

    Like maybe the tables used by the step??

    """

    def __init__(self, name, wrapped):
        self.name = name
        self.func = _get_callable(wrapped)

    def __call__(self, **local_kwargs):
        kwargs = _collect_inputs(self.func, **local_kwargs)
        result = self.func(**kwargs)
        _notify_changed(self.name)
        return result


class ColumnWrapper(object):
    """
    Wraps a pandas.Series or a callable that returns one.

    """

    def __init__(self, name, wrapped, clear_on=None, attach_to=None, arg_map={}):

        # TODO: if cached, make sure attachments are in clear events?

        # create the injectable
        self.name = name
        self._injectable = _create_injectable(name, wrapped, clear_on, arg_map=arg_map)

        # add attachments
        _attach(name, attach_to)

    def __call__(self):
        result = self._injectable()
        assert isinstance(result, pd.Series)
        result.name = self.name
        return result


class TableWrapper(object):
    """
    Wraps a pandas.DataFame or a callable that returns one.

    """

    def __init__(self, name, wrapped,
                 clear_on=None, attach_to=None, columns=None, arg_map={}):

        # create the injectable
        self.name = name
        self._injectable = _create_injectable(name, wrapped, clear_on, arg_map=arg_map)
        self._local_columns = columns

        # add attachments
        _attach(name, attach_to)

    def __call__(self):
        """
        For collection/evaluation, return the wrapper.
        TODO: return some type of view instead?

        """
        return self

    @property
    def local(self):
        """
        Returns the local (source) data frame.

        """
        df = self._injectable()
        assert isinstance(df, pd.DataFrame)
        self._local_columns = df.columns
        return df

    @property
    def index(self):
        """
        Returns the table's index.

        """
        return self.local.index

    def __len__(self):
        """
        Return the number of rows in the local data frame.

        """
        return len(self.local)

    @property
    def local_columns(self):
        """
        Returns columns that are apart of the local data frame.

        """
        if self._local_columns is None:
            # if column names aren't cached, force the injectable
            # to be evaluated
            self._local_columns = self.local.columns
        return list(self._local_columns)

    @property
    def attached_columns(self):
        """
        Return the NAMES of the attached columns.

        """
        cols = []

        if self.name in _attachments:
            attached = _attachments[self.name]

            for a in attached:
                if a not in _injectables:
                    continue

                inj = _injectables[a]

                if isinstance(inj, ColumnWrapper):
                    cols.append(inj.name)
                elif isinstance(inj, TableWrapper):
                    # do we want to attach all columns or just local?
                    # for now, get everything
                    cols += inj.columns

        return cols

    @property
    def columns(self):
        """
        Names of all columns, both local and attached.

        """
        return self.local_columns + self.attached_columns

    def get_column(self, column_name):
        """
        Fetches a column from the wrapper.

        """
        # 1st look locally
        if column_name in self.local_columns:
            return self.local[column_name]

        # now look to attachments
        if self.name in _attachments:
            attached = _attachments[self.name]

            for a in attached:

                # fetch the injectable
                if a not in _injectables:
                    continue
                inj = _injectables[a]

                # columns
                if isinstance(inj, ColumnWrapper):
                    if column_name == inj.name:
                        return inj()

                # tables
                if isinstance(inj, TableWrapper):
                    if column_name in inj.columns:
                        return inj[column_name]

        raise ValueError("column '{}' not found in table '{}'".format(column_name, self.name))

    def __getitem__(self, key):
        """"
        Returns a column via <wrapper>['<column_name>'] syntax.

        """
        return self.get_column(key)

    def __getattr__(self, key):
        """
        Returns a column via <wrapper>.<column_name> syntax.

        """
        return self.get_column(key)

    def to_frame_all(self, copy_local=True):
        """
        Return a view of the table wrapper with ALL local and attached columns.

        """
        df = self.local
        df_concat = [df]
        series_concat = []

        # get attached
        if self.name in _attachments:

            attached = _attachments[self.name]

            for a in attached:

                # fetch the injectable
                if a not in _injectables:
                    continue
                inj = _injectables[a]

                # columns
                if isinstance(inj, ColumnWrapper):
                    series_concat.append(inj())

                # tables
                elif isinstance(inj, TableWrapper):
                    # do we want to attach all columns or just local?
                    # for now, let's stick with just local
                    df_concat.append(inj.to_frame_all(False))

        # concat independent columns into a data frame
        if len(series_concat) > 0:
            df_concat.append(pd.concat(series_concat, axis=1))

        # concat attached tables
        if len(df_concat) > 1:
            df = pd.concat(df_concat, axis=1)
        else:
            # no attachments, just copy the local
            if copy_local:
                df = df.copy()

        return df

    def to_frame(self, columns=None, copy_local=True):
        """
        Returns a view of the table as a dataframe. All values
        in the data frame are copies.

        """
        # if columns are not provided, return everything
        if not columns:
            return self.to_frame_all()

        # treat the desired columns as a set
        if not isinstance(columns, list):
            columns = [columns]
        col_set = set(columns)

        df_concat = []
        series_concat = []

        # first check local
        df = self.local
        from_local = set(df.columns) & col_set
        if len(from_local) > 0:
            df = df[list(from_local)]
            if copy_local:
                df = df.copy()
            df_concat = [df]
        col_set -= from_local

        # now scan attachments
        if self.name in _attachments:
            attached = _attachments[self.name]

            for a in attached:

                # break if everything has been found
                if len(col_set) == 0:
                    break

                # fetch the injectable
                if a not in _injectables:
                    continue
                inj = _injectables[a]

                # columns
                if isinstance(inj, ColumnWrapper):
                    if inj.name in col_set:
                        series_concat.append(inj())
                        col_set.remove(inj.name)

                # tables
                elif isinstance(inj, TableWrapper):
                    curr_cols = col_set & set(inj.columns)
                    if len(curr_cols) > 0:
                        df_concat.append(inj.to_frame(list(curr_cols), False))
                        col_set -= curr_cols

        if len(col_set) > 0:
            raise ValueError('Columns: {}, not found'.format(col_set))

        # concat independent columns into a data frame
        if len(series_concat) > 0:
            df_concat.append(pd.concat(series_concat, axis=1))

        # concat attached tables
        if len(df_concat) == 0:
            raise ValueError("Problem in to_frame")
        final_df = pd.concat(df_concat, axis=1)

        # return the final data frame with columns in the desired order
        return final_df[columns]

    def update_col(self, column_name, series):
        """
        Add or replace a column in the underlying DataFrame.
        If local is a function result, this will be overriden
        after the next collection.

        """
        self.local[column_name] = series

        # notify changes, assume event named <table_name>.<column_name>
        _notify_changed('{}.*'.format(self.name))
        _notify_changed('{}.{}'.format(self.name, column_name))

    def __setitem__(self, key, value):
        self.update_col(key, value)

    def update_col_from_series(self, column_name, series, cast=False):
        """
        Update existing values in a column from another series.
        Index values must match in both column and series. Optionally
        casts data type to match the existing column.

        Parameters
        ---------------
        column_name : str
        series : panas.Series
        cast: bool, optional, default False

        """

        col_dtype = self.local[column_name].dtype
        if series.dtype != col_dtype:
            if cast:
                series = series.astype(col_dtype)
            else:
                err_msg = "Data type mismatch, existing:{}, update:{}"
                err_msg = err_msg.format(col_dtype, series.dtype)
                raise ValueError(err_msg)

        self.local.loc[series.index, column_name] = series
        _notify_changed('{}.{}'.format(self.name, column_name))


########################
# PUBLIC MODULE-LEVEL
########################

def clear_all():
    """
    Re-initializes everything.

    """
    _init_globals()


def clear_cache():
    """
    Clears the cache.

    """
    _notify_changed('env')


def add_injectable(name, wrapped, clear_on=None, autocall=True, arg_map={}):
    """
    Adds a wrapped value or function as an injectable.

    """
    _injectables[name] = _create_injectable(name, wrapped, clear_on, autocall, arg_map)
    _notify_changed(name)


def add_column(name, wrapped, attach_to=None, clear_on=None, arg_map={}):
    _injectables[name] = ColumnWrapper(name, wrapped, clear_on, attach_to, arg_map)
    _notify_changed(name)


def add_table(name, wrapped, attach_to=None, clear_on=None, columns=None, arg_map={}):
    _injectables[name] = TableWrapper(
        name, wrapped, clear_on, attach_to, columns, arg_map)
    _notify_changed(name)


def add_step(name, wrapped):
    """
    Adds a step to the environment.

    """
    _injectables[name] = StepFuncWrapper(name, wrapped)


def get_injectable(name):
    """
    Returns an injectable.

    """
    if name in _injectables:
        return _injectables[name]
    return None


def eval_injectable(name, **kwargs):
    """
    Evaluates an injectable and returns the result.

    """
    inj = get_injectable(name)
    if inj is not None:
        return inj(**kwargs)
    return None


def list_injectables():
    """
    Return the names of the injected items.

    """
    return _injectables.keys()


def list_attachments():
    """
    Return registered attachements. Keys are the table names the items
    are attached to, values are the names of the attached items'.

    """
    return _attachments


def list_tables():
    """
    Returns the names of registered tables.

    """
    return [t.name for t in
            filter(lambda t: isinstance(t, TableWrapper), _injectables.values())]


def list_steps():
    """
    Returns the names of registered steps.

    """
    return [s.name for s in
            filter(lambda s: isinstance(s, StepFuncWrapper), _injectables.values())]


def load_tables(hdf_file, prefix='*', basenames=None, remove_prefix=False):
    """
    Opens an hdf file and registers all the tables with orca.

    Parameters:
    -----------
    hdf_file: string,
        Full path to the hdf file containing the tables.
    prefix: str, optional default *
        Prefix to filter in the tables to load. By default load everything.
    basenames: list of str, optional, default None
        Indicates specific tables to load, e.g. ['households', 'buildings'].
    remove_prefix: boolm optional, default False
        If True, the prefix will be removed from the table name when registering,
        e.g. '/2020/households' is registered as 'households'.
        Note: it might be dangerous to remove the prefix if a strict prefix filter is
        not provided as there may be multiple tables with the same basename.

    TODO: maybe look into some regex functions to clean up the logic?

    """
    # evaluate and format the prefix filter
    strict_prefix = True
    prefix = str(prefix)

    if not prefix.startswith('/'):
        # treat 2010 the same as /2010
        prefix = '/' + prefix

    if prefix.endswith('*'):
        # handle wildcard filters
        strict_prefix = False
        prefix = prefix[:len(prefix) - 1]

    if len(prefix) > 1 and prefix.endswith('/'):
        # treat 2010 the same as 2010/
        prefix = prefix[:len(prefix) - 1]

    with pd.get_store(hdf_file, mode='r') as store:

        for t in store.keys():
            t_base = os.path.basename(t)
            t_dir = os.path.dirname(t)
            do_load = True

            # check the prefix
            if strict_prefix:
                if prefix != t_dir:
                    do_load = False
            else:
                if not t.startswith(prefix):
                    do_load = False

            # check the base tables
            if basenames is not None and t_base not in basenames:
                do_load = False

            # load in the table
            if do_load:
                orca_name = t

                if remove_prefix:
                    orca_name = t_base

                add_table(orca_name, store[t])


def write_tables(fname, table_names=None, prefix=None, write_attached=False):
    """
    Writes tables to a pandas.HDFStore file.

    Parameters
    ----------
    fname : str
        File name for HDFStore. Will be opened in append mode and closed
        at the end of this function.
    table_names: list of str, optional, default None
        List of tables to write. If None, all registered tables will
        be written.
    prefix: str
        If not None, used to prefix the output table names so that
        multiple iterations can go in the same file.
    write_attached: bool, optional, default False
        If True, all columns are written out. If False, only the
        local columns will be written.

    """
    if table_names is None:
        table_names = list_tables()

    tables = (get_injectable(t) for t in table_names)
    key_template = '{}/{{}}'.format(prefix) if prefix is not None else '{}'

    with pd.get_store(fname, mode='a') as store:
        for t in tables:
            if write_attached:
                store[key_template.format(t.name)] = t.to_frame()
            else:
                store[key_template.format(t.name)] = t.local


def run(steps, iter_vars=None, data_out=None, out_interval=1,
        out_base_tables=None, out_run_tables=None, write_attached=False):
    """
    Runs a sequence of steps. Mostly the same as the orca run
    method?

    A step can be a single entry or a tuple
    in the form (name, iter_vars), where iter_vars is the
    subset of years to run

    Note: right now, the tables must be specified to be writted,
    tables are not inferred from the steps.

    """

    # clear out run cache
    _notify_changed('run')

    # write out the base
    if data_out and out_base_tables:
        write_tables(data_out, out_base_tables, 'base', write_attached)

    # run the steps
    iter_vars = iter_vars or [None]
    max_i = len(iter_vars)

    for i, var in enumerate(iter_vars, start=1):

        print 'on iteration: {}'.format(var)
        iter_time = time.time()

        # update iteration variables
        _notify_changed('iteration')
        add_injectable('iter_var', var)

        # execture the step sequence
        for s in steps:
            curr_step = s

            # ignore if the step is not available
            # for the current iteration
            if isinstance(s, tuple):
                if var not in s[1]:
                    continue
                curr_step = s[0]

            print '--executing: {}'.format(curr_step)
            step_time = time.time()

            # update step variables
            _notify_changed('step')

            # execute the step
            eval_injectable(curr_step)

            print '---time: {:.2f} s'.format(time.time() - step_time)

        # write out results
        if data_out and out_run_tables:
            if (i - 1) % out_interval == 0 or i == max_i:
                write_tables(data_out, out_run_tables, var, write_attached)

        print '---time: {:.2f} s'.format(time.time() - iter_time)


########################
# DECORATORS
# NEED TO THINK ABOUT HOW
# TO MAKE IT EASIER TO DEFINE
# CACHE CLEARING DEPENDENCIES??
########################


def get_name(name, func):
    if name:
        return name
    else:
        return func.__name__


def injectable(name=None, clear_on=None):
    """
    Decorates functions that will register
    a generic injectable.

    """
    def decorator(func):
        add_injectable(get_name(name, func), func, clear_on)
        return func
    return decorator


def callback(name=None):
    """
    Decorates functions that will return a callback function.

    """
    def decorator(func):
        add_injectable(get_name(name, func), func, autocall=False)
        return func
    return decorator


def column(name=None, attach_to=None, clear_on=None):
    """
    Decorates functions that will register
    a column

    """
    def decorator(func):
        add_column(get_name(name, func), func, attach_to, clear_on)
        return func
    return decorator


def table(name=None, attach_to=None, clear_on=None, columns=None):
    """
    Decorates functions that will register
    a table

    """
    def decorator(func):
        add_table(get_name(name, func), func, attach_to, clear_on, columns)
        return func
    return decorator


def step(name=None):
    """
    Decorates functions that will register
    a step.

    """
    def decorator(func):
        add_step(get_name(name, func), func)
        return func
    return decorator


################################
# BROADCASTING / RELATIONSHIPS
################################
