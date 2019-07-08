"""
Working on a major update to orca.

"""

from __future__ import print_function

try:
    from inspect import getfullargspec as getargspec
except ImportError:
    from inspect import getargspec


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
    spec = getargspec(func)
    args = spec.args
    defaults = spec.defaults

    # get keyword args for the function's default values
    default_kwargs = {}
    if defaults is not None:
        kw_start_idx = len(args) - len(defaults)
        default_kwargs = dict(zip([key for key in args[kw_start_idx:]], list(defaults)))

    return args, default_kwargs

