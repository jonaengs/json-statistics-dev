import os
import types
import inspect
from munch import Munch, munchify

from compute_structures import PruneStrat, StatType

# munchify converts the dictionary to an object, similar to in Javascript. Allows dot notation for member accesss
settings = munchify({
    "logger": {
        "quiet": False,
        "out_dir": "logs/",
    },
    "stats": {
        "stat_type": StatType.HISTOGRAM,
        "hyperloglog_error": 0.05,
        "filename": "mini",

        "force_new": False,
        "data_dir": "data/recsys/",
        "out_dir": "stats/",

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
        # "prune_params": [
        #     {
        #         "strat": PruneStrat.MIN_FREQ,
        #         "threshold": 0.01
        #     },
        #     {
        #         "strat": PruneStrat.MAX_NO_PATHS,
        #         "threshold": 100
        #     },
        #     {
        #         "strat": PruneStrat.PRUNE_PREFIX,
        #         "threshold": 3
        #     },
        # ],
    },
    "tracking": {
        "print_tracking": False,

        "local_memory": False,
        "global_memory": True,
        "time": True,
    }
})


def make_property(obj, attr):
    """ Makes the given attribute a property of the object, as if the property decorator had been applied to it """
    # For a more general solution that works for object that don't support the dictionart lookup syntax,
    #   change obj[attr] to getattr(obj, attr)
    setattr(obj, attr, types.MethodType(obj[attr], obj))
    setattr(obj.__class__, attr, obj[attr])
    setattr(obj.__class__, attr, property(obj[attr]))

def auto_make_properties(parent):
    """ Traverse the Munch object. Transform callables with a self argument into properties """
    for key, child in parent.items():
        if type(child) == Munch:
            auto_make_properties(child)
        elif callable(child) and inspect.getfullargspec(child)[0][0] == 'self':
            make_property(parent, key)

# make_property(settings.stats, "out_path")
# make_property(settings.stats, "data_path")
auto_make_properties(settings)

if __name__ == '__main__':
    assert settings.stats.out_path == (os.path.join(settings.stats.out_dir, settings.stats.filename) + ".json")
    assert settings.stats.data_path == (os.path.join(settings.stats.data_dir, settings.stats.filename) + ".json")

    print(settings.stats.out_path)
    settings.stats.filename = "adasdasd"
    print(settings.stats.out_path)

    # import yaml
    # print(yaml.safe_dump(settings))
