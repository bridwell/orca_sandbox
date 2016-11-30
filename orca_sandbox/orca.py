"""
Trying out an over-hauled verion of orca, as a potential
simulation orchestration framework.

This is very experimental.

"""


from orca_sandbox.events import *
from orca_sandbox.collector import *


_injectables = {}
_events = init_events(['clear_all', 'run', 'iteration', 'step'])


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
        Function to register.
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


def _do_collect(func):
    """
    Collects inputs required by the function,
    given the current collection of injectables.

    """
    return collect_inputs(func, _injectables)


class InjectableWrapper(Collectable):
    """

    """

    def __init__(self, name, wrapped, cache=False, clear_on=None, autocall=True):
        # set internals
        self.name = name
        self.cache = cache
        if clear_on is not None:
            self.cache = True
        self.autocall = autocall
        self.update(wrapped)

        # register cache clearing events
        _register_clear_events(self.clear_cached, clear_on)

    def update(self, wrapped):
        """
        Updates the wrapped function or value. Notifies changes.

        """
        if inspect.isfunction(wrapped):
            # a function is wrapped
            self.func = wrapped
            self.data = None

        elif callable(wrapped):
            # a class with __call__ is wrapped, need to create an instance first
            kwargs = _do_collect(wrapped.__init__)
            self.func = wrapped(kwargs)
            self.data = None

        else:
            # a value is wrapped
            self.func = None
            self.data = wrapped
            self.cache = None

        # notify the change in state
        _notify_changed(self.name)

    def collect(self, *kwargs):
        """
        Member of Collectable.
        Called when variables are collected.

        """
        if self.autocall:
            return self.__call__(**kwargs)
        else:
            return self

    def __call__(self, **local_kwargs):
        """
        Returns the wrapped value or function results.

        """
        if self.func is not None and ((not self.cache) or self.data is None):
            # evaluate the function
            kwargs = _do_collect(self.func, **local_kwargs)
            self.data = self.func(**kwargs)
            _notify_changed(self.name)

        # return the results
        return self.data

    def clear_cached(self):
        """
        Clears out the cached data. Notifies subscribers.

        """
        if self.func is not None:
            self.data = None


def add_injectable(name, wrapped, cache=False, clear_on=None):
    """
    Adds an injectable.

    """
    inj = InjectableWrapper(name, wrapped, cache, clear_on)
    _injectables[name] = inj
    return inj


def get_injectable(name):
    """
    Returns an injectable.

    """
    if name in _injectables:
        return _injectables[name]
    return None


def eval_injectable(name):
    """
    Collects and injectable.

    """
    inj = get_injectable(name)
    if inj is not None:
        return inj.collect()
    return None


def list_injectables():
    return _injectables.keys()


class ColumnWrapper(InjectableWrapper):
    """
    Wraps either a pandas.Series or function that returns one.

    Also support a numpy array with this?

    """

    def __init__(self, name, wrapped, cached=False, clear_on=None):
        # init from the super
        super(ColumnWrapper, self).__init__(
            name, wrapped, cached, clear_on)

        # shortcut our evaluation calls to the super do I need this?
        self.local = super(ColumnWrapper, self).__call__


class DataFrameWrapper(InjectableWrapper):
    """
    Wraps a pandas.DataFrame or a function that returns one.

    """

    class DataFrameWrapper(InjectableWrapper):

        def __init__(self, name, wrapper,
                     cached=False, clear_on=None, column_names=None):

            # init from the super
            super(DataFrameWrapper, self).__init__(
                name, wrapped, cached, clear_on, autocall=False)

            # shortcut our evaluation calls to the super do I need this?
            self.local = super(DataFrameWrapper, self).__call__