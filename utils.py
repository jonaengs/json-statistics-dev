import psutil
from pprint import pprint

def print_memory_use():
    info = psutil.Process().memory_info()
    # print("RSS:", info.rss//(1024**2), info.rss//1024, info.rss)
    # print("VMS:", info.vms//(1024**2), info.vms//1024, info.vms)

    print(f"RSS used: {info.rss/(1024**2):.1f} MB")