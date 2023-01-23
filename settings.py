from munch import munchify

from compute_structures import StatType

# munchify converts the dictionary to an object, similar to in javascript. Allows dot notation for member accesss
settings = munchify({
    "logger": {
        "quiet": False,
        "out_dir": "logs/",
    },
    "stats": {
        "stat_type": StatType.HISTOGRAM,
        "hyperloglog_error": 0.05,
        "filename": "test",

        "force_new": True,
        "data_folder": "data/recsys/",
        "out_folder": "stats/",
    },
    "tracking": {
        "print_tracking": False,

        "local_memory": False,
        "global_memory": True,
        "time": True,
    }
})