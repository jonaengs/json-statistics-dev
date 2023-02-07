from copy import deepcopy
import os
import types
import inspect
from munch import Munch, munchify

from compute_structures import PruneStrat, StatType
from utils import deep_dict_update

# TODO: Make sure everything works with lists of dictionaries/munch objects

_settings_are_locked = False

# munchify converts the dictionary to an object, similar to in Javascript. Allows dot notation for member accesss
settings = munchify({
    "logger": {
        "store_output": True,  # redirect output to devnull, so that nothing is stored
        "silenced": False,
        "out_dir": "logs/",
    },
    "stats": {
        "stats_type": StatType.HISTOGRAM,
        "force_new": False,
        "sampling_rate": 0.0,
        "hyperloglog_error": 0.05,

        "filename": "mini",
        "data_dir": "data/recsys/",
        "out_dir": "stats/",
        "stats_cache_dir": "stats/cache/",

        "data_path": lambda self, *_: os.path.join(self.data_dir, self.filename) + ".json",
        "out_path": lambda self, *_: os.path.join(self.out_dir, self.filename) + ".json",

        "prune_strats": [PruneStrat.MIN_FREQ],
        "prune_params": {
            PruneStrat.MIN_FREQ: {
                "threshold": 0.01
            },
            PruneStrat.MAX_NO_PATHS: {
                "threshold": 100
            },
            PruneStrat.MAX_PREFIX_LENGTH: {
                "threshold": 3
            },
        },
    },
    "tracking": {
        "print_tracking": False,

        "local_memory": False,
        "global_memory": True,
        "time": True,
    }
})


def _locked(method):
    def wrapper(*args, **kwargs):
        if not _settings_are_locked:
            return method(*args, **kwargs)
        if _settings_are_locked and _LockableMunch.err_on_attempt:
            raise ValueError("Object is locked. Values cannot be set.")

    return wrapper

class _LockableMunch(Munch):
    err_on_attempt = False

    @_locked
    def __setitem__(self, __key, __value) -> None:
        return super().__setitem__(__key, __value)

    @_locked
    def __setattr__(self, k, v):
        return super().__setattr__(k, v)

    @_locked 
    def __delattr__(self, k):
        return super().__delattr__(k)

    @_locked
    def __setstate__(self, state):
        return super().__setstate__(state)


def unlock_settings():
    """
    Makes the settings object mutable again.
    Beware: Turns all tuples into lists.
    """
    global _settings_are_locked

    def rec_apply_unlock(parent):
        for key, child in parent.items():
            if isinstance(child, Munch):
                rec_apply_unlock(child)
            elif type(child) == tuple:
                parent[key] = list(child)

    rec_apply_unlock(settings)
    _settings_are_locked = False
    

def lock_settings(err_on_attempt=True):
    """
    Makes the settings object immutable.
    Beware: turns lists into tuples. 
    TODO? Make a custom immutable list class, overriding (__set, pop, __del, ...)
    """
    global _settings_are_locked
            
    def rec_apply_lock(parent):
        for key, child in parent.items():
            if isinstance(child, Munch):
                rec_apply_lock(child)
            elif type(child) == list:
                parent[key] = tuple(child)

        parent.__class__ = _LockableMunch


    rec_apply_lock(settings)
    _settings_are_locked = True
    _LockableMunch.err_on_attempt = err_on_attempt


def update_settings(data: dict) -> None:
    deep_dict_update(settings, data, modification_only=True)

def _make_property(obj, attr):
    """ Makes the given attribute a property of the object, as if the property decorator had been applied to it """
    # For a more general solution that works for object that don't support the dictionart lookup syntax,
    #   change obj[attr] to getattr(obj, attr)
    setattr(obj, attr, types.MethodType(obj[attr], obj))  # Make attr a method
    setattr(obj.__class__, attr, obj[attr])  # Make attr a method of the class
    setattr(obj.__class__, attr, property(obj[attr]))  # Make the method a property

def _auto_make_properties(parent):
    """ Traverse the Munch object. Transform callables with a self argument into properties """
    for key, child in parent.items():
        if isinstance(child, Munch):
            _auto_make_properties(child)
        elif callable(child) and inspect.getfullargspec(child)[0][0] == 'self':
            _make_property(parent, key)

# _make_property(settings.stats, "out_path")
# _make_property(settings.stats, "data_path")
_auto_make_properties(settings)


###  TEST FUNCS

def _rec_test_lock(parent):
    for key, child in parent.items():
        if isinstance(child, Munch):
            _rec_test_lock(child)
            parent[key] = tuple(child)

        parent[key] = 1
        assert parent[key] == child
        parent[key] = None
        assert parent[key] == child

        if hasattr(parent, str(key)):
            setattr(parent, str(key), 1)
            getattr(parent, str(key)) == child
            setattr(parent, str(key), None)
            getattr(parent, str(key)) == child

        if type(child) == list:  # Does nothing right now, as all lists are made into tuples
            prev_len = len(child)
            child.append(None)
            assert len(parent[key]) == prev_len == len(child) - 1

            child = child + [key]
            assert parent[key] != child 


def _rec_test_unlock(parent):
    for key, child in parent.items():
        if isinstance(child, Munch):
            _rec_test_unlock(child)
            parent[key] = tuple(child)

        child_copy = deepcopy(child)

        parent[key] = 1
        assert parent[key] == 1
        parent[key] = None
        assert parent[key] == None

        child_is_property = hasattr(type(parent), str(key)) and isinstance(getattr(type(parent), str(key)), property)
        if not child_is_property and hasattr(parent, str(key)):
            setattr(parent, key, 1)
            assert parent[key] == 1
            setattr(parent, key, None)
            assert parent[key] == None

        if type(child) == list:
            child.append(None)
            assert parent[key][-1] == None
            child += [key]
            assert child[-1] == child 

        parent[key] = child_copy


if __name__ == '__main__':
    assert settings.stats.out_path == (os.path.join(settings.stats.out_dir, settings.stats.filename) + ".json")
    assert settings.stats.data_path == (os.path.join(settings.stats.data_dir, settings.stats.filename) + ".json")

    # print(settings.stats.out_path)
    # settings.stats.filename = "adasdasd"
    # print(settings.stats.out_path)

    # import yaml
    # print(yaml.safe_dump(settings))

    lock_settings(err_on_attempt=False)
    _rec_test_lock(settings)
    unlock_settings()
    _rec_test_unlock(settings)

    # Make sure we didn't fuck up the settings somehow by locking and unlocking the settings
    assert settings.stats.out_path == (os.path.join(settings.stats.out_dir, settings.stats.filename) + ".json")
    assert settings.stats.data_path == (os.path.join(settings.stats.data_dir, settings.stats.filename) + ".json")
