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
    Adds a wrapped value or function as an injectable.

    """

    def create_injectable():
        if not callable(wrapped):
            return ValueWrapper(name, wrapped)
        if not autocall:
            return CallbackWrapper(name, wrapped)
        if clear_on:
            return CachedFuncWrapper(name, wrapped, clear_on)
        else:
            return FuncWrapper(name, wrapped)

    _injectables[name] = create_injectable()
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


# DON'T USE THIS FOR NOW, USE ADD INJECTABLE INSTEAD
# def update_injectable(name, wrapped):
#    """
#    Updates an injectable.

#    """

#    inj = get_injectable(name)
#    if inj is not None:
#        inj.update(wrapped)


def list_injectables():
    return _injectables.keys()


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


def _collect_and_eval(name, func, **local_kwargs):
    """
    Collects inputs and evluates the function.

    """
    kwargs = _do_collect(func, **local_kwargs)
    result = func(**kwargs)
    _notify_changed(name)
    return result


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
