"""
An event organizes calls to a set of subscribing
functions. This based on the Observer pattern and is
similar to the event/delegate architecture in c#.

To subscribe to an event:
my_event += my_func

To unsubscribe:
my_event -= my_func

To raise/fire the event:
my_event()

Optionally an argment matching function can be provided
that generates the arguments to be sent to subscribers.
Othewrise, the subscribers must have the same signature as
the event call.

For example firing:
my_event(10)

# then the subscribing function should have
my_func(iter_var):
    ...

Based on:

http://stackoverflow.com/questions/1092531/event-system-in-python
http://www.valuedlessons.com/2008/04/events-in-python.html

"""


class Event(object):
    """
    Used to assign functions or handlers to a common event.

    Paramaters:
    -----------
    arg_match_func: function, optional, default None:
        Callback used to provide argument matching
        when firing a callback. If not provided the
        args of the event call must match all of the
        handlers.

    """

    def __init__(self, arg_match_func=None):
        # initialize the event
        self.subscribers = set()
        self.arg_match_func = arg_match_func

    def __iadd__(self, subscriber):
        # adds a subsriber (or listener) to the event
        self.subscribers.add(subscriber)
        return self

    def __isub__(self, subscriber):
        # removes a subscriber (or listener) from the event
        if subscriber in self.subscribers:
            self.subscribers.remove(subscriber)
        return self

    def __call__(self, *args, **kwargs):
        # when the event fires, execute all subscribing functions
        # if the event has an argument matching function defined
        # then use this
        for subscriber in self.subscribers:
            if self.arg_match_func is None:
                subscriber(*args, **kwargs)
            else:
                matched_args = self.arg_match_func(subscriber, **kwargs)
                subscriber(**matched_args)

    def clear(self):
        # remove all subscribers
        self.subscribers = set()


def subscribe_to_events(events, subscribe_to, func, arg_match_func=None):
    """
    Helper function that registers a function with one or more
    events. Generates new events if not already existing
    in the dictionary. Modfies the dictionary in replace and returns
    nothing.

    Parameters:
    -----------
    events: Dictionary<str, Event>
        Dictionary containing all events.
    subscribe_to: str or list of str
        Events to subscribe to.
    func:
        The function to subscribe/register.
    arg_match_func: optional, default None
        Argument matching function to use when firing events.

    """

    # do nothing if events are invalid or func is not callable
    if events is None:
        return
    if subscribe_to is None:
        return
    if not callable(func):
        return

    # subscribes to single event, if the event does not exist
    # yet then create it
    def subscribe(event_name):
        if event_name in events:
            e = events[event_name]
        else:
            e = Event(arg_match_func)
            events[event_name] = e
        e += func

    # subscribes to all
    if isinstance(subscribe_to, list):
        for event_name in subscribe_to:
            subscribe(event_name)
    else:
        subscribe(subscribe_to)


def init_events(defaults, arg_match_func=None):
    """
    Creates a dictionary of events for the provided defaults.

    Parameters:
    ----------
    defaults: list of str
        Default events to register
    arg_match_func: optional, default None
        Argument matching function to use when firing events

    Returns:
    -------
    Dictionary of events.

    """
    events = {}
    for d in defaults:
        events[d] = Event(arg_match_func)
    return events
