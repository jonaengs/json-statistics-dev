from copy import deepcopy
import os
import types
import inspect
import typing
from munch import Munch, munchify

from compute_structures import PruneStrat, StatType
from utils import deep_dict_update

# TODO: Make sure everything works with lists of dictionaries/munch objects

_settings_are_locked = False

# munchify converts the dictionary to an object, similar to in Javascript. Allows dot notation for member accesss
settings = munchify({
    "logger": {
        "store_output": False,  # If False, logger will redirect output to devnull, creating no files
        "silenced": False,
        "out_dir": "logs/",
    },
    "stats": {
        "stats_type": StatType.HISTOGRAM,
        "force_new": True,
        "sampling_rate": 0.0,
        "hyperloglog_error": 0.05,
        "num_histogram_buckets": 25,

        "singleton_plus_enabled": True,
        "enum_statistics_enabled": True,

        "key_path_key_sep": '.',
        "key_path_type_sep": '_',

        "data_dir": "data/",
        "data_source": "recsys",
        "filename": "mini",
        "data_path": lambda self, *_: os.path.join(self.data_dir, self.data_source, self.filename) + ".json",
        
        "out_dir": "stats/",
        "out_path": lambda self, *_: os.path.join(self.out_dir, self.filename) + ".json",
        "stats_cache_dir": "stats/cache/",


        "prune_strats": [PruneStrat.MIN_FREQ],
        "prune_params": {
            "min_freq_threshold": 0.001,
            "max_no_paths_threshold": 100,
            "max_prefix_length_threshold": 3,
        },
    },
    "tracking": {
        "print_tracking": False,

        "local_memory": False,
        "global_memory": True,
        "time": True,
    }
})

def update_settings(data: dict) -> None:
    deep_dict_update(settings, data, modification_only=True)

def lock_settings():
    """
    Makes the settings object immutable.
    Beware that this is kinda !@?#$!$, as it does a mix of changing the class of the objects 
    themselves (all Munch instances for example), and replacing instances whose class cannot be
    changed dynamically with locked copies.
    So sometimes the return value in the recursive traversal is important, and sometimes it is not.
    """            
    def rec_apply_lock(parent):
        if isinstance(parent, str):  # Skip strings to avoid endless recursion
            return parent

        if isinstance(parent, typing.Iterable):
            for key, child in (parent.items() if isinstance(parent, dict) else enumerate(parent)):
                locked_child = rec_apply_lock(child)
                parent[key] = locked_child

        return _change_into_locked_class(parent)

    global _settings_are_locked
    rec_apply_lock(settings)
    _settings_are_locked = True


def unlock_settings():
    """
    Makes the settings object mutable again.
    """
    def rec_apply_unlock(parent):
        if isinstance(parent, str): 
            return parent

        if isinstance(parent, typing.Iterable):
            for key, child in (parent.items() if isinstance(parent, dict) else enumerate(parent)):
                parent[key] = rec_apply_unlock(child)
        
        return parent.get_original() if isinstance(parent, _Lockable) else parent

    global _settings_are_locked
    _settings_are_locked = False
    rec_apply_unlock(settings)

def _locked(method, err_on_attempt=False):
    def wrapper(*args, **kwargs):
        if not _settings_are_locked:
            return method(*args, **kwargs)
        if _settings_are_locked and err_on_attempt:
            raise ValueError("Object is locked. Values cannot be set.")

    return wrapper

class _Lockable:
    pass

def _make_lockable_subclass(super_class):
    class Lockable_(super_class, _Lockable):
        def get_original(self):
            try:
                self.__class__ = super_class
                return self
            except TypeError:
                return super_class(self)

        def __init__(self, *args, **kwargs):
            return super().__init__(*args, **kwargs)

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

        @_locked
        def append(self, *args, **kwargs):
            return super().append(*args, **kwargs)

        @_locked
        def pop(self, *args, **kwargs):
            return super().pop(*args, **kwargs)

        def __deepcopy__(self, *args, **kwargs): 
            return super_class(self)

    Lockable_.__name__ = Lockable_.__name__ + super_class.__name__
    return Lockable_

def _change_into_locked_class(val_to_lock):
    """
    Returns the value with its class changed to a subclass of the value's original class, 
    but with all (?) mutating methods locked. 
    If the value is not mutable, the value is returned unchanged.
    """
    
    def is_hashable(v):
        # Simple test, but does not cover cases where the hash method throws an error
        if not isinstance(v, typing.Hashable):
            return False
        try: 
            hash(v)  # Better test. Fails for things like 'def __hash__(self): raise ValueError("Not hashable")'
        except:
            return False
        return True

    # Use hashability as a proxy for immutability
    is_immutable = is_hashable(val_to_lock)
    if is_immutable:
        return val_to_lock


    base_class = type(val_to_lock)
    locked_class = _make_lockable_subclass(base_class)

    try:
        val_to_lock.__class__ = locked_class
        return val_to_lock
    except TypeError:
        # Cannot assign __class__ on built-in types
        # Instead, try to make a subclass and pass in the original's value
        locked_val_copy = locked_class(val_to_lock)
        return locked_val_copy


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
    def dict_tests(parent, key, child):
        parent[key] = 1
        assert parent[key] == child
        parent[key] = None
        assert parent[key] == child

    def setattr_tests(parent, key, child):
        setattr(parent, str(key), 1)
        getattr(parent, str(key)) == child
        setattr(parent, str(key), None)
        getattr(parent, str(key)) == child

    def list_tests(parent, key, child: list):
        prev_len = len(child)
        child.append(None)
        assert len(parent[key]) == prev_len

        child = child + [key]
        assert parent[key] != child 

    if isinstance(parent, typing.Iterable):
        for key, child in (parent.items() if isinstance(parent, dict) else enumerate(parent)):
            if not isinstance(child, str):
                _rec_test_lock(child)
            
            dict_tests(parent, key, child)

            child_is_property = hasattr(type(parent), str(key)) and isinstance(getattr(type(parent), str(key)), property)
            if not child_is_property and hasattr(parent, str(key)):
                setattr_tests(parent, key, child)

            if isinstance(child, list):
                list_tests(parent, key, child)



def _rec_test_unlock(parent):
    def dict_tests(parent, key):
        parent[key] = 1
        assert parent[key] == 1
        parent[key] = None
        assert parent[key] == None

    def setattr_tests(parent, key):
        setattr(parent, key, 1)
        assert parent[key] == 1
        setattr(parent, key, None)
        assert parent[key] == None

    def list_tests(child: list):
        child.append(1)
        assert child[-1] == 1
        child.append(2)
        assert child[-1] == 2

        child.pop()
        assert child[-1] == 1

        prev_child_0 = child[0]
        child[0] = not child[0]
        assert child[0] == (not prev_child_0)

    if isinstance(parent, typing.Iterable):
        for key, child in (parent.items() if isinstance(parent, dict) else enumerate(parent)):
            child_copy = deepcopy(child)
            if not isinstance(child, str):
                _rec_test_unlock(child)
            
            dict_tests(parent, key)

            child_is_property = hasattr(type(parent), str(key)) and isinstance(getattr(type(parent), str(key)), property)
            if not child_is_property and hasattr(parent, str(key)):
                setattr_tests(parent, key)

            if isinstance(child, list):
                list_tests

            parent[key] = child_copy


if __name__ == '__main__':
    assert settings.stats.out_path == (os.path.join(settings.stats.out_dir, settings.stats.filename) + ".json")
    assert settings.stats.data_path == (os.path.join(settings.stats.data_dir, settings.stats.data_source, settings.stats.filename) + ".json")

    # print(settings.stats.out_path)
    # settings.stats.filename = "adasdasd"
    # print(settings.stats.out_path)

    # import yaml
    # print(yaml.safe_dump(settings))

    lock_settings()
    _rec_test_lock(settings)
    unlock_settings()
    _rec_test_unlock(settings)

    # Make sure we didn't fuck up the settings somehow by locking and unlocking the settings
    assert settings.stats.out_path == (os.path.join(settings.stats.out_dir, settings.stats.filename) + ".json")
    assert settings.stats.data_path == (os.path.join(settings.stats.data_dir, settings.stats.data_source, settings.stats.filename) + ".json")
