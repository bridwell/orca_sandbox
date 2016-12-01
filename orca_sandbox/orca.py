"""
Trying out an over-hauled verion of orca, as a potential
simulation orchestration framework.

This is very experimental.

"""

import inspect
import pandas as pd

from .events import *
from .collector import *


########################
# PRIVATE MODULE-LEVEL
########################

def _init_globals():
    """
    Initializes module-level variables.

    """
    global _injectables, _events, _attachments
    _injectables = {}
    _attachments = {}
    _events = init_events(['clear_all', 'run', 'iteration', 'step'])


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
    if name in _events:
        _events[name]()


def _register_clear_events(func, clear_on):
    """
    Registers cache clearing events. A 'clear_all'
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
        clear_on = ['clear_all']
    if isinstance(clear_on, str):
        clear_on = [clear_on]
    if 'clear_all' not in clear_on:
        clear_on = ['clear_all'] + clear_on

    subscribe_to_events(_events, clear_on, func, collect_inputs)


def _do_collect(func, **kwargs):
    """
    Collects inputs required by the function,
    given the current collection of injectables.

    """
    return collect_inputs(func, _injectables, **kwargs)


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
        init_kwargs = _do_collect(wrapped.__init__)
        obj = wrapped(**init_kwargs)
        return obj.__call__
    else:
        raise ValueError('The wrapped argument must be a function or callable class.')


def _collect_and_eval(name, func, **local_kwargs):
    """
    Collects inputs and evluates the function.

    """
    kwargs = _do_collect(func, **local_kwargs)
    result = func(**kwargs)
    _notify_changed(name)
    return result


def _create_injectable(name, wrapped, clear_on=None, autocall=True):
    """
    Creates an injectable from a provided value or function.

    """

    if not callable(wrapped):
        return ValueWrapper(name, wrapped)
    if not autocall:
        return CallbackWrapper(name, wrapped)
    if clear_on:
        return CachedFuncWrapper(name, wrapped, clear_on)
    else:
        return FuncWrapper(name, wrapped)


def _attach(name, attach_to):
    """
    Links an injectable with other injectables it it attached to. Used to
    bind columns or tables with other tables.

    """

    if attach_to is None:
        return

    def attach(target_name):
        if target_name in _attachments:
            a = _attachments[target_name]
        else:
            a = set()
            _attachments[target_name] = a
        a.add(name)

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
    Wraps a function.

    For the moment, leave this not cacheable.

    """

    def __init__(self, name, wrapped):
        self.name = name
        self.func = _get_callable(wrapped)

    def __call__(self, **local_kwargs):
        return _collect_and_eval(self.name, self.func, **local_kwargs)


class CachedFuncWrapper(FuncWrapper):
    """
    Wraps a function that supports caching.

    """

    def __init__(self, name, wrapped, clear_on):

        # init from FuncWrapper
        super(CachedFuncWrapper, self).__init__(name, wrapped)

        # set up caching
        self._data = None
        _register_clear_events(self.clear_cache, clear_on)

    def __call__(self, **local_kwargs):
        """
        TODO: figure out what to do if we have local_kwargs,
        does this not get applied for cahcing??

        ALSO, think about how this interacts with a global cache
        (on or off) setting.

        """
        if self._data is None:
            self._data = _collect_and_eval(self.name, self.func, **local_kwargs)

        return self._data

    def clear_cache(self):
        self._data = None


class ColumnWrapper(object):
    """
    Wraps a pandas.Series or a callable that returns one.

    """

    def __init__(self, name, wrapped, clear_on=None, attach_to=None):

        # TODO: if cached, make sure attachments are in clear events?

        # create the injectable
        self._injectable = _create_injectable(name, wrapped, clear_on)

        # add attachments
        _attach(name, attach_to)

    def __call__(self):
        result = self._injectable()
        assert isinstance(result, pd.Series)
        return result


class TableWrapper(object):
    """
    Wraps a pandas.DataFame or a callable that returns one.

    """

    def __init__(self, name, wrapped,
                 clear_on=None, attach_to=None, columns=None):

        # create the injectable
        self._injectable = _create_injectable(name, wrapped, clear_on)
        self._local_columns = columns

        # add attachments
        # do this later, right now just attach columns
        # _attach(name, attach_to)

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
    def attached_tables(self):
        """
        Returns names of attached tables.

        """
        return []

    @property
    def attached_columns(self):
        """
        Return names of attached columns.

        """
        return []

    @property
    def columns(self):
        """
        Names of all columns, both local and attached.

        """
        return self.local_columns + self.attached_columns

    def get_column(self, column_name):
        """
        Fetches a column from the wrapper.

        Right now, this just returns the local

        """
        return self.local[column_name]

    def __getitem__(self, key):
        return self.get_column(key)

    # def __getattr__(self, key):
    #    return self.get_column(key)

    def to_frame(self, columns=None):
        """
        Returns a view of the table as a dataframe.

        """
        return

    def update_col(self, column_name, series):
        """
        Add or replace a column in the underlying DataFrame.
        If local is a function result, this will be overriden
        after the next collection.

        """
        self.local[column_name] = series

    def __setitem__(self, key, value):
        """
        Updates or adds a column with syntax:

        wrapper['col'] = series

        """
        return self.update_col(key, value)

    # def __setattr__(self, key, value):
    #    """
    #    Updates or adds a columns with syntax:
    #
    #    wrapper.col1 = series
    #
    #    """
    #   return self.update_col(key, value)

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


########################
# PUBLIC MODULE-LEVEL
########################

def clear_all():
    """
    Re-initializes everything.

    """
    _init_globals


def clear_cache():
    """
    Clears the cache.

    """
    _notify_changed('clear_all')


def add_injectable(name, wrapped, clear_on=None, autocall=True):
    """
    Adds a wrapped value or function as an injectable.

    """
    _injectables[name] = _create_injectable(name, wrapped, clear_on, autocall)
    _notify_changed(name)


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
    return _injectables.keys()


def add_column(name, wrapped, attach_to=None, clear_on=None):
    # TODO: do we need to un-register attachments?

    _injectables[name] = ColumnWrapper(name, wrapped, clear_on, attach_to)
    _notify_changed(name)


def add_table(name, wrapped, attach_to=None, clear_on=None, columns=None):
    # TODO: do we need to un-register attachments?

    _injectables[name] = TableWrapper(
        name, wrapped, clear_on, attach_to, columns)
    _notify_changed(name)
