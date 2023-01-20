from munch import munchify

from compute_structures import StatType

# munchify converts the dictionary to an object, similar to in javascript. Allows dot notation for member accesss
settings = munchify({
    "logger": {
        "quiet": False
    },
    "stats": {
        "stat_type": StatType.HISTOGRAM,
        "hyperloglog_error": 0.05,
        "filename": "test",

        "data_folder": "data/recsys/",
        "out_folder": "stats/",
    },
    "tracking": {
        "peak_memory": False,
        "time": False,
    }
})