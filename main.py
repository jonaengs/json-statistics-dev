import sys
import argparse
from compute_structures import StatType


from settings import settings


def update_settings(args):
    for path, value in args.__dict__.items():
        target = settings
        keys = path.split("-")
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value


if __name__ == '__main__':
    # NOTE: Take care to update settings before importing any other files!
    # Can't really have dots in argument names, so use dashes instead
    parser = argparse.ArgumentParser()
    parser.add_argument("stats-filename", nargs="?", default="test", help="name of file to use as data src (training or test)")
    parser.add_argument("-q", "--quiet", dest="logger-quiet", action="store_true", help="toggle printing to stdout")
    parser.add_argument("-s", "--stat_type", choices=(StatType._member_names_), dest="stats-stat_type")
    # parser.add_argument("-s", "--stat_type", dest="stats-stat_type", choices=[e.value for e in StatType], type=int)
    
    args = parser.parse_args()
    if "stats-stat_type" in args: 
        setattr(args, "stats-stat_type", StatType[getattr(args, "stats-stat_type")])
    
    update_settings(args)


    import logger
    
    logger.log(" ".join(sys.argv))
    logger.log(args)

    import compute
    compute.run()