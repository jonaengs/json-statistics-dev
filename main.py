import sys
import argparse
import random

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


def update_settings(namespace: argparse.Namespace, exclude_list=[]):
    for path, value in vars(namespace).items():
        if path in exclude_list:
            continue

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

    # Arguments that touch settings
    parser.add_argument("stats.filename", nargs="?", help="name of file to use as data src (training or test)")
    parser.add_argument("-d", "--data-dir", dest="stats.data_source")
    parser.add_argument("-q", "--quiet", dest="logger.silenced", action="store_true", help="stop logger printing to stdout")
    parser.add_argument("-n", "--new", dest="stats.force_new", action="store_true", help="Force creation of fresh statistics")
    parser.add_argument("-s", "--stats_type", dest="stats.stats_type", choices=StatType._member_names_, action=EnumAction.create(StatType))
    parser.add_argument("-p", "--prune_strats", dest="stats.prune_strats", choices=PruneStrat._member_names_, nargs="*", action=EnumAction.create(PruneStrat))

    # Arguments that don't touch settings
    parser.add_argument("-m", "--mode", choices=["a", "v"], nargs="*", help="Which path to execute. Can do multiple. Separate with space.", default=[])

    args = parser.parse_args()
    update_settings(args, exclude_list=["mode"])

    settings.logger.store_output = True

    # Do initial logging
    import logger
    logger.log(" ".join(sys.argv), quiet=True)
    logger.log(args, quiet=True)
    logger.logger.log_settings()


    # BEGIN PROGRAM
    import analyze
    if not args.mode or "a" in args.mode:
        # Seed the random module
        seed = random.randrange(0, sys.maxsize)
        # seed = 8481792292457891019
        random.seed(seed)
        logger.log("random seed:", seed)

        analyze.run_analysis()
    if "v" in args.mode:
        analyze.examine_analysis_results()