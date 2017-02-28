"""
Another take at trying to enhance orca to allow:

- Re-usable functions via an argmap.
- Dynamic evaluation of caches.
- Attaching data frames AND columns.
- Better handling of broadcasts and relationships across tables.

"""

import inspect
import pandas as pd

########################
# PRIVATE MODULE-LEVEL
########################


def _init_globals():
    """
    Initializes module-level variables.

    """
    global _injectables, _clear_events, _attachments

    # store all injectables in a single dictionary, therefore names must be unique
    _injectables = {}

    # stores the injectables along with the events that can clear their cache
    _clear_events = {
        'clear_all': set(),
        'run': set(),
        'iteration': set(),
        'step': set()
    }

    # stored relationshipes between injectables, keys are the name of the destination
    # injectable, values are sets of the linked injectable (i.e. columns)
    _attachments = {}


# initialize the globals upon importing
# is there a better way to do this?
_init_globals()


def _clear_caches(event_name):
    """
    Clears out the caches for the provided event.

    """
    if event_name not in _clear_events:
        raise ValueError('Event {}, does not exit'.format(event_name))

    for inj_name in _clear_events[event_name]:
        _injectables[inj_name].clear()


def _get_func_args(func):
    """
    Returns a function's argument names and default values. These are used by other
    functions to establish dependencies and collect inputs.

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


def _collect_inputs(func, requester, arg_map={}):
    """
    Collects the inputs from the environment that are needed to execute the provided function.

    ** Still not sure what to do with defaults, I guess use these as expressions? **

    Parameters:
    -----------
    func: callable:
        The function callable to execute.

    requester: str
        Name of the injectable doing the collection.

    arg_map: dict, optional, default {}
        Dictionary that maps between the argument names of the function (keys) and the
        corresponding injectables (values). This allows for re-using the same function
        with different injected inputs.

        For example:
            def my_func(a):
                ...
            arg_map = {'a': 'my_injectable'}

        Would collect the injectable named 'my_injectable' to be used for the 'a' argument.
        If no arg_map is provided then the injectable named 'a' would be collected for
        the 'a' argument.

    Returns:
    --------
    - Named keyword arg dictionary containing needed to execute the function.
    - Bool indicating if the provided function can use a previously cached value or if it needs
        to be re-evaluated, this is based on if the any of the required inputs have been
        re-evaluated since they were last collected.
            - True indicates that a cached value may be re-used.
            - False indicates that the function needs to be re-evaluated.

    """
    kwargs = {}
    cache = True

    # get function signature
    arg_names, defaults = _get_func_args(func)

    # if no args are needed by the function then we're done
    if len(arg_names) == 0:
        return kwargs, cache

    # loop the through the function args and find the matching input
    for a in arg_names:

        if a == 'self':
            # call coming from a class, ignore
            continue

        # fetch the injectable
        name = a
        if a in arg_map:
            name = arg_map[a]
        if name not in _injectables:
            # argument not found
            raise ValueError("Injectable {} not found".format(name))
        inj = _injectables[name]

        # do the collection
        if not isinstance(inj, _AbstractWrapper):
            raise ValueError("Injectable {} not based on AbstractWrapper".format(name))
        kwargs[a], curr_cache = inj.collect(requester)
        if not curr_cache:
            cache = False

    return kwargs, cache


def _create_injectable(name, wrapped, autocall=True, cache_scope=None, arg_map={}):
    """
    Factory method to create an instance of an injectable. Note: this isn't actually
    adding anthing to the environment. See add_injectable method for that.

    """
    if not callable(wrapped):
        return _ValueWrapper(name, wrapped)

    if not autocall:
        return _CallbackWrapper(name, wrapped)

    if cache_scope is None:
        return _FuncWrapper(name, wrapped, arg_map)

    else:
        use_collect_status = False
        if cache_scope == 'inputs':
            use_collect_status = True
        return _CachedFuncWrapper(name, wrapped, use_collect_status, arg_map)


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


class _AbstractWrapper(object):
    """
    Abstract class for wrappers. TODO: enfore this with ABC.

    """

    def clear(self):
        """
        Clears and cached information.

        """
        pass

    def collect(self, requester):
        """
        Evaluates and returns the injectable for the given requester.
        Should return a tuple in the form result, cache_status
            - Result: the result of the collection
            - cache_status: bool True if a cached value was returned, False
                if a new value is returned since the last call from the requesing
                injectable.

        """
        pass


class _ValueWrapper(_AbstractWrapper):
    """
    Wraps a value.

    """

    def __init__(self, name, value):
        self.name = name
        self._data = value
        self.clear()

    def clear(self):
        """
        Clears out cached dependents. Not sure if I really need this?

        """
        self._cached = set()

    def collect(self, requester):
        """
        Returns the wrapped value, and notifies the requesting injectable if it has
        been provided before.

        """
        if requester in self._cached:
            return self._data, True
        else:
            self._cached.add(requester)
            return self._data, False


class _CallbackWrapper(_AbstractWrapper):
    """
    Wraps a callback function that can be injected into another function.

    """
    def __init__(self, name, wrapped):
        self.name = name
        self._wrapped = wrapped

    def clear(self):
        pass

    def collect(self, requester):
        return self._wrapped, False


class _FuncWrapper(_AbstractWrapper):
    """
    Wraps a function that does NOT support caching.

    """

    def __init__(self, name, wrapped, arg_map={}):
        self.name = name
        self._wrapped = wrapped
        self._arg_map = arg_map

    def clear(self):
        """
        Just implemented to support the Abstract. Not needed.

        """
        pass

    def collect(self, requester):
        """
        Do the evaluation.

        """
        collected, _ = _collect_inputs(self._wrapped, self._arg_map)
        results = self._wrapped(collected)
        return results, False


class _CachedFuncWrapper(_AbstractWrapper):
    """
    Wraps a function that supports caching.

    """

    def __init__(self, name, wrapped, use_collect_status=False, arg_map={}):
        self.name = name
        self._wrapped = wrapped
        self._arg_map = arg_map
        self.use_collect_status = use_collect_status
        self.clear()

    def clear(self):
        """
        Clears out cached data.

        """
        self._cached = set()
        self._data = None

    def collect(self, requester):

        # collect the inputs
        # remember:
        #   cache_status of True indicate the input has not change since the last collection
        #   cache_status of False indicates one or more of the inputs has changed
        collected, cache_status = _collect_inputs(self._wrapped, self._arg_map)

        # determine if we need to invalidate the cache
        if self.use_collect_status and cache_status is False:
            self.clear()

        # evaluate the function if necessary
        if self._data is None:
            self._data = self.wrapped(collected)

        # return the result and the cache status for the given requster
        if requester in self.cached:
            return self._data, True
        else:
            self._cached.add(requester)
            return self.data, False


class _ColumnWrapper(_AbstractWrapper):
    """
    Wraps a pandas.Series or a callable that returns one.

    """

    def __init__(self, name, wrapped, cache_scope=None, attach_to=None, arg_map={}):

        # create the injectable
        self.name = name
        self._injectable = _create_injectable(name, wrapped, True, cache_scope, arg_map)

        # add attachments
        _attach(name, attach_to)

    def clear(self):
        self._injectable.clear()

    def collect(self, requester):
        result, cache_status = self._injectable.collect(requester)
        assert isinstance(result, pd.Series)
        result.name = self.name
        return result, cache_status


class _TableWrapper(_AbstractWrapper):
    """
    Wraps a pandas.DataFame or a callable that returns one.

    """

    def __init__(self, name, wrapped, cache_scope=None, attach_to=None, columns=None, arg_map={}):

        # create the injectable
        self.name = name
        self._injectable = _create_injectable(name, wrapped, True, cache_scope, arg_map)
        self._local_columns = columns

        # add attachments
        _attach(name, attach_to)


    def clear(self):
        # maintain a cache for each column?
        self._cache = {}  # this will be a dictionary of sets


    def collect(self, requester):
        """
        For collection/evaluation, return the wrapper.
        TODO: return some type of view instead?

        """
        return self


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
    Clears all caches.

    """
    _clear_caches('clear_all')


def add_injectable(name, wrapped, autocall=True, cache_scope=None, arg_map={}):
    """
    Creates and adds an injectable to the environment.

    """

    inj = _create_injectable(name, wrapped, autocall, cache_scope, arg_map)
    _injectables[name] = inj

    # set up clear events
    _clear_events['clear_all'].add(name)

    if cache_scope in _clear_events.keys():
        _clear_events[cache_scope].add(name)


"""
def add_column(name, wrapped, attach_to=None, clear_on=None, arg_map={}):
    _injectables[name] = ColumnWrapper(name, wrapped, clear_on, attach_to, arg_map)
    _notify_changed(name)
"""

"""
def add_table(name, wrapped, attach_to=None, clear_on=None, columns=None, arg_map={}):
    _injectables[name] = TableWrapper(
        name, wrapped, clear_on, attach_to, columns, arg_map)
    _notify_changed(name)
"""


########################
# DECORATORS
########################


def get_name(name, func):
    if name:
        return name
    else:
        return func.__name__


def injectable(name=None, autocall=True, cache_scope=None):
    """
    Decorates functions that will register
    a generic injectable.

    """
    def decorator(func):
        add_injectable(get_name(name, func), func, autocall, cache_scope)
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
