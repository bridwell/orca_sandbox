from ..events import *


def test_event():
    """
    Test the general sequence of creating
    events, wiring/un-wiring methods to them
    and firing them.

    """

    # set up some functions to wire
    results = [1, 1]

    def double():
        results[0] *= 2

    def quadruple():
        results[1] *= 4

    # create the event and wire the functions
    e = Event()
    e += double
    e += quadruple

    # test firing both
    e()
    assert results[0] == 2
    assert results[1] == 4

    # remove one and re-fire
    e -= double
    e()
    assert results[0] == 2
    assert results[1] == 16

    # clear out everything and re-fire
    e.clear()
    e()
    assert results[0] == 2
    assert results[1] == 16


def test_event_fire_with_args():
    """
    Test event firing when providing arguments.

    In these cases no callback function is given
    to the event so the signature of the event fire call
    must match the signature of all the subscribers.

    """
    results = [10]

    def multiply_by(other):
        results[0] *= other

    e = Event()
    e += multiply_by
    e(5)
    assert results[0] == 50


def test_event_fire_with_args_with_callback():
    """
    Test event firing when providing arguments.

    Here a callback method is used to resolve the
    parameters that are sent to the subscribers.

    """

    # use this to check results
    results = [0, 0, 0]

    # define some functions
    def func1():
        results[0] += 1

    def func2(other_val):
        results[1] += other_val

    def func3(other_val2):
        results[2] += other_val2

    # define a simple callback that resolve the
    # arguments to be passed to each subscribing method
    def get_args(func, **kwargs):

        name = func.__name__
        out_kwargs = {}

        if name == 'func2':
            out_kwargs['other_val'] = kwargs['other_val']
        elif name == 'func3':
            out_kwargs['other_val2'] = kwargs['other_val2']

        return out_kwargs

    # create, wire, fire
    e = Event(get_args)
    e += func1
    e += func2
    e += func3
    e(other_val2=100, other_val=20)

    # inspect the results
    assert results[0] == 1
    assert results[1] == 20
    assert results[2] == 100


def test_subscribe_to_events():

    events = {}
    results = [0, 0]

    def my_func1():
        results[0] = 20

    def my_func2():
        results[1] += 100

    def my_func3():
        results[1] += 10

    # calls with invalid inputs should not generate events
    subscribe_to_events(None, ['a', 'b'], None)
    assert len(events) == 0

    subscribe_to_events(events, None, None)
    assert len(events) == 0

    subscribe_to_events(events, ['a', 'b'], 10)
    assert len(events) == 0

    # subscribe to a single event
    subscribe_to_events(events, 'a', my_func1)
    assert len(events) == 1
    events['a']()
    assert results[0] == 20

    # subscribe to multiple events
    subscribe_to_events(events, ['b', 'c'], my_func2)
    events['b']()
    assert results[1] == 100
    events['c']()
    assert results[1] == 200

    # subscribe to an already existing event
    subscribe_to_events(events, ['c'], my_func3)
    events['c']()
    assert results[1] == 310
