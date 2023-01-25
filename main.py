import sys
import argparse

from compute_structures import StatType
from settings import settings

class EnumAction(argparse.Action):
    def __init__(self, enum, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        
        super().__init__(option_strings, dest, **kwargs)
        self.enum = enum

    def __call__(self, _parser, namespace, value, _option_string=None):
        setattr(namespace, self.dest, self.enum[value])

    @staticmethod
    def create(enum):
        return lambda *a, **k: EnumAction(enum, *a, **k)


def update_settings(args: argparse.Namespace):
    for path, value in vars(args).items():
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
    
    parser.add_argument("stats.filename", nargs="?", default="test", help="name of file to use as data src (training or test)")
    parser.add_argument("-q", "--quiet", dest="logger.quiet", action="store_true", help="toggle printing to stdout")
    parser.add_argument("-n", "--new", dest="stats.force_new", action="store_true", help="Force creation of fresh statistics")
    parser.add_argument("-s", "--stat_type", dest="stats.stat_type", choices=(StatType._member_names_), action=EnumAction.create(StatType))
    
    args = parser.parse_args()    
    update_settings(args)


    import logger
    logger.log(" ".join(sys.argv), quiet=True)
    logger.log(args, quiet=True)

    import compute
    compute.run()