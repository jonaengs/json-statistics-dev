import sys
import argparse

from compute_structures import StatType, PruneStrat
from settings import settings

class EnumAction(argparse.Action):
    def __init__(self, enum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enum = enum

    def __call__(self, _parser, namespace, values, _option_string=None):
        if type(values) == list:
            setattr(namespace, self.dest, [self.enum[val] for val in values])
        else:
            val = values
            setattr(namespace, self.dest, self.enum[val])

    @staticmethod
    def create(enum):
        return lambda *a, **k: EnumAction(enum, *a, **k)


def update_settings(namespace: argparse.Namespace):
    for path, value in vars(namespace).items():
        if value is not None:
            keys = path.split(".")

            target = settings
            for key in keys[:-1]:
                target = target[key]

            assert keys[-1] in target, f"setting key '{keys[-1]}' not present in settings object {target}. (Key-chain={keys})"
            target[keys[-1]] = value


if __name__ == '__main__':
    # NOTE: Take care to update settings before importing any other files!
    parser = argparse.ArgumentParser()
    
    parser.add_argument("stats.filename", nargs="?", help="name of file to use as data src (training or test)")
    parser.add_argument("-q", "--quiet", dest="logger.silenced", action="store_true", help="stop logger printing to stdout")
    parser.add_argument("-n", "--new", dest="stats.force_new", action="store_true", help="Force creation of fresh statistics")
    parser.add_argument("-s", "--stats_type", dest="stats.stats_type", choices=(StatType._member_names_), action=EnumAction.create(StatType))
    parser.add_argument("-p", "--prune_strats", dest="stats.prune_strats", choices=(PruneStrat._member_names_), nargs="*", action=EnumAction.create(PruneStrat))
    
    args = parser.parse_args()    
    update_settings(args)
    
    settings.logger.store_output = True
    import logger
    logger.log(" ".join(sys.argv), quiet=True)
    logger.log(args, quiet=True)
    logger.logger.log_settings()

    # import compute_stats
    # compute_stats.run()

    import analyze
    # analyze.run_analysis()
    analyze.examine_analysis_results()