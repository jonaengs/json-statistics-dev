
import json
from settings import settings

_cache = {}
def load_data():
    data_fname = settings.stats.filename

    if data_fname not in _cache:
        with open(settings.stats.data_path) as f:
            _cache[data_fname] = json.load(f)

    assert len(_cache) == 1, _cache.keys()
    return _cache[data_fname]

    