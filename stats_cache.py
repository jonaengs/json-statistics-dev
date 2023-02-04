

from collections import namedtuple
from copy import deepcopy
import os
import pickle
import time
from compute_structures import StatType
from settings import settings

"""
Pickles statistics objects and stores them to disk.
Associates each pickled object with the settings active
at the time of pickling. If the current program settings 
match any stored settings, the associated pickled stats
object will be unpickled and returned. 
"""

# Use an index to associate pickled stats files and setting configurations
IndexEntry = namedtuple("IndexEntry", ("file_name", "stat_settings"))
_cache_index: list[IndexEntry] = None

def add_stats(stats):
    """
    Store a stat object to disk and add it to the index using the currently active settings.
    Will delete any previous stat objects with the same settings they exist.
    """
    global _cache_index

    assert stats is not None
    assert stats[1]["stats_type"] == settings.stats.stats_type, (stats[1]["stats_type"], settings.stats.stats_type)

    # Store the stat object to file
    stat_fname = str(time.time_ns())[4:] + ".pickle"  # stat fname can be anything, as long as it is unique
    stat_pickle_path = os.path.join(settings.stats.stats_cache_dir, stat_fname)
    open(stat_pickle_path, mode="x").close()  # Error if file already exists
    with open(stat_pickle_path, mode="wb") as f:
        pickle.dump(obj=stats, file=f)
        f.flush()


    # Delete any other file with the same settings
    current_stat_settings = _get_identifying_settings()
    to_remove = [fname for fname, f_settings in _cache_index if f_settings == current_stat_settings]
    for fname in to_remove:
        fpath = os.path.join(settings.stats.stats_cache_dir, fname)
        os.remove(fpath)

    _cache_index = [entry for entry in _cache_index if not entry.file_name in to_remove]

    # Add the newly stored stats to the index and write the updated index to disk
    _cache_index.append(IndexEntry(stat_fname, current_stat_settings))
    _store_index()

def get_cached_stats():
    """
    Returns the stats object stored with the same stat settings as currently active.
    Raises en error if the object can't be found.
    """
    search_res = _get_cached_stats_file()
    if search_res:
        return _load_cached(search_res)
    
    raise ValueError("Could not find matching cached stats file")

def check_cached_stats_exists():
    """Returns True if settings matching the current settings are stored in the index. False otherwise"""
    current_stat_settings = _get_identifying_settings()
    return any(
        cached_stat_settings ==  current_stat_settings
        for (_f_name, cached_stat_settings) in _cache_index
    )


##########################################################################################

def _setup():
    """Load _cache_index from file"""
    global _cache_index

    cache_index_fname = "index.pickle"
    cache_dir = settings.stats.stats_cache_dir
    cache_index_path = os.path.join(cache_dir, cache_index_fname)

    # Create stats cache dir if it doesn't exist yet
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    # Try to load cache index from file
    try:
        with open(cache_index_path, mode="rb") as f:
            _cache_index = pickle.load(f)
    except FileNotFoundError:
        _cache_index = []


def _load_cached(stat_fname):
    """ Return the cached stats object stored in the file with the given name """
    stat_pickle_path = os.path.join(settings.stats.stats_cache_dir, stat_fname)
    with open(stat_pickle_path, mode="rb") as f:
        return pickle.load(f)

def _get_cached_stats_file() -> str | None:
    current_stat_settings = _get_identifying_settings()
    for (f_name, stat_settings) in _cache_index:
        if stat_settings == current_stat_settings:
            return f_name

def _get_identifying_settings():
    """Get a deepcopy of the settings that identify a stats object, in a pickleable format"""
    def rec_del_lambdas(parent: dict):
        for key, child in parent.items():
            if isinstance(child, dict):  # Munch subclasses dict, so instances of it will pass this test
                rec_del_lambdas(child)
            elif callable(child):
                parent[key] = None

    stat_settings = deepcopy(settings.stats)
    # Remove all lambdas from the object, as they can't be pickled
    rec_del_lambdas(stat_settings)

    return stat_settings


def _store_index():
    """Write _cache_index to file"""

    cache_index_fname = "index.pickle"
    cache_index_path = os.path.join(settings.stats.stats_cache_dir, cache_index_fname)
    with open(cache_index_path, mode="wb") as f:
        pickle.dump(_cache_index, f)
        f.flush()






if __name__ != '__main__':
    _setup()
else:  
    # if __name__ == '__main__'

    # Setup fake file system so tests don't affect real data
    cwd = os.getcwd()
    settings.stats.stats_cache_dir = settings.stats.out_dir + "test_cache/"

    from pyfakefs import fake_filesystem
    filesystem = fake_filesystem.FakeFilesystem()
    open = fake_filesystem.FakeFileOpen(filesystem)
    os = fake_filesystem.FakeOsModule(filesystem)

    filesystem.create_dir(cwd)
    os.chdir(cwd)
    filesystem.create_dir(settings.stats.stats_cache_dir)

    cache_index_fname = "index.pickle"
    cache_index_path = os.path.join(settings.stats.stats_cache_dir, cache_index_fname)


    # BEGIN TESTS:
    
    # No cache index should exist on disk yet. Test stats cache dir should be empty
    assert not os.path.exists(cache_index_path)
    assert not os.listdir(settings.stats.stats_cache_dir)

    _setup()

    # No cached stats should exist yet
    print(len(_cache_index))
    assert check_cached_stats_exists() == False
    try:
        get_cached_stats()
        assert False
    except ValueError:
        pass

    # _cache_index should be empty
    assert len(_cache_index) == 0
    assert os.path.exists


    # Create a stats object, and add it to the cache. Make sure everything works as expected
    from compute_stats import make_statistics
    import data_cache

    collection = data_cache.load_data()
    stats = make_statistics(collection)
    add_stats(stats)

    # Stats file was created
    assert check_cached_stats_exists() == True
    stats_from_cache = get_cached_stats()
    assert stats_from_cache is not None
    # Stats file content matches expected content
    assert stats_from_cache == stats

    # _cache_index was updated and stored to disk
    assert len(_cache_index) == 1
    with open(cache_index_path, 'rb') as f:
        assert pickle.load(f) == _cache_index

    # Settings stored match current settings
    assert _cache_index[0].stat_settings == _get_identifying_settings()


    # Changing current settings ensures nothing matches
    old_stat_type = settings.stats.stats_type
    settings.stats.stats_type = StatType((settings.stats.stats_type.value + 1) % len(StatType))
    assert _cache_index[0].stat_settings != _get_identifying_settings()
    assert check_cached_stats_exists() == False
    # Changing back works as expected
    settings.stats.stats_type = old_stat_type
    assert _cache_index[0].stat_settings == _get_identifying_settings()
    assert check_cached_stats_exists() == True

    # Writing with same settings does not result in more files
    add_stats(stats)
    add_stats(stats)
    add_stats(stats)
    assert len(_cache_index) == 1
    assert len(os.listdir(settings.stats.stats_cache_dir)) == 2  # index plus single cached obj
    


    