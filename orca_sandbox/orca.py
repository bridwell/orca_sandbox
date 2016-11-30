"""
Trying out an over-hauled verion of orca, as a potential
simulation orchestration framework.

This is very experimental.

"""


from .events import *
from .collector import *


########################
# PRIVATE MODULE-LEVEL
########################

def _init_globals():
    """
    Initializes module-level variables.

    """
    global _injectables, _events
    _injectables = {}
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


def _do_collect(func, **kwargs):
    """
    Collects inputs required by the function,
    given the current collection of injectables.

    """
    return collect_inputs(func, _injectables, **kwargs)


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


def add_injectable(name, wrapped, cache=False, clear_on=None, autocall=True):
    """
    Adds an injectable.

    """
    inj = InjectableWrapper(name, wrapped, cache, clear_on, autocall)
    _injectables[name] = inj


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
        return inj.collect()
    return None


def update_injectable(name, wrapped):
    """
    Updates an injectable.

    """

    inj = get_injectable(name)
    if inj is not None:
        inj.update(wrapped)


def list_injectables():
    return _injectables.keys()


########################
# INJECTABLE CLASSES
########################


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
            init_kwargs = _do_collect(wrapped.__init__)
            obj = wrapped(**init_kwargs)
            self.func = obj.__call__
            self.data = None

        else:
            # a value is wrapped
            self.func = None
            self.data = wrapped
            self.cache = None

        # notify the change in state
        _notify_changed(self.name)

    def collect(self, **kwargs):
        """
        Member of Collectable.
        Called when variables are collected.

        """
        if self.autocall:
            return self.__call__(**kwargs)
        else:
            # should this be the function instead?
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
