"""
Provides methods for collecting inputs and argments needed by functions
that are 'managed'. Managed here refers to functions and global variables
that are registered within, and called by, some type of orchestration
framework rather than called directly.

This is an extension of (and heavily influenced by) the variable collection
mechansims in https://github.com/UDST/orca.


See the tests for examples.

"""
import inspect


class Collectable(object):
    """
    Classes inheriting from Collectable will be evaluated upon
    being collected. Override the collect method for custom
    behavior.

    """

    def collect(self):
        """
        Evaluates an injected callable and returns the result.

        Overide this method to provide additional or different
        logic when being collected.

        """
        return self.__call__()


def collect_inputs(func, injectables={}, **local_kwargs):
    """
    For a given function and call, resolves and collects the input parameters needed.

    This considers 3 types of inputs, resolved with the following priority:

        1 - Local parameters, i.e. those passed in from a calling function.
            - These arguments must be named. Non-named args will be ignored.
            - Arguments passed that are not defined by the function being called will
              also be ignored.
                - This allows the calling function (e.g. a fired event) to overload
                    calls to subscribing functions.

        2 - Global variables or injectables, similar to what is done with orca.

        3 - Function defaults, if no local or globals are found, then defaults
        defined by the called function will be used

    Parameters:
    -----------
    func: Callable
        The function being called
    injectables: Dictionary, optional
        Dictionary of global injectables. Typcially these are managed by
        the orchestration framework.
    **local_kwargs:
        Key word arguments from a call.

    Returns:
    --------
    Named keyword arg dictionary that can be passed to execute the function.

    """
    kwargs = {}

    # get the called function's args/parameters
    spec = inspect.getargspec(func)
    args = spec.args
    defaults = spec.defaults

    # if no args are needed by the function then we're done
    if len(args) == 0:
        return kwargs

    # get keyword args for the function's default values
    default_kwargs = {}
    if defaults is not None:
        kw_start_idx = len(args) - len(defaults)
        default_kwargs = dict(zip([key for key in args[kw_start_idx:]], list(defaults)))

    # loop the through the function args and find the matching input
    for a in args:

        if a == 'self':
            continue

        # local inputs (i.e. keywords from the calling function)
        if a in local_kwargs:
            kwargs[a] = local_kwargs[a]

        # global inputs injectables
        elif a in injectables:
            inj = injectables[a]

            # this needs to get removed,
            # leave-in so we don't break tests
            # todo: update the tests
            if isinstance(inj, Collectable):
                inj = inj.collect()

            # temporary -- use for re-factored wrappers
            elif callable(inj):
                inj = inj()

            kwargs[a] = inj

        # function defaults
        elif a in default_kwargs:
            kwargs[a] = default_kwargs[a]

        else:
            # argument not found, throw an exception
            # alternatively this could set the value to None?
            raise ValueError("Argument {} not found".format(a))

    return kwargs
