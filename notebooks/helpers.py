# Requires 'sys.path.append("..")' in importing file
from math import log2
from pyparsing import Literal
from compute_structures import *
import matplotlib.pyplot as plt

def dedup(l: list):
    out = []
    for e in l:
        if e not in out:
            out.append(e)
    return out


def scatterplot(x, y, point_data, ylabel="", xlabel="", ):
    markers_signify: Literal["sampling_rate", "prune_strats"] | None = "sampling_rate"

    allowed_markers = ['o', 'v', 'P', '*', 'X']
    colors = [setting["stats_type"].value for setting in point_data]
    sizes = [
        100 + 60*max((log2(setting.get("num_histogram_buckets", 2)) - 3), 0)
        for setting in point_data
    ]

    if markers_signify == "sampling_rate":
        sample_ratio_to_marker_map = dict(zip(sorted(set(p["sampling_rate"] for p in point_data)), allowed_markers))
        markers = [sample_ratio_to_marker_map[p["sampling_rate"]] for p in point_data]
    elif markers_signify == "prune_strats":
        for pd in point_data:
            pd["prune_strats"] = tuple(pd["prune_strats"])

        prune_strat_to_marker_map = dict(zip(set(p["prune_strats"] for p in point_data), allowed_markers))
        markers = [prune_strat_to_marker_map[p["prune_strats"]] for p in point_data]
    else:
        markers = []

    fig, ax = plt.subplots(tight_layout=True, figsize=[6, 4])
    scp = ax.scatter(x, y, c=colors, s=sizes)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)



    ## LEGENDS ##
    # Thanks for helping: https://stackoverflow.com/a/43814479

    # Legend 1: Stats type / Color
    leg1 = plt.legend(
        handles=scp.legend_elements()[0], 
        labels={setting["stats_type"].name: 0 for setting in point_data}.keys(),
        loc="upper right", 
        title="Statistics Type")
    ax.add_artist(leg1)

    # Legend 2: Various / Marker
    if markers:
        h = [
            plt.plot([],[], marker=marker, ls="None", color="grey")[0]
            # for i, marker in enumerate(set(markers))]
            for i, marker in enumerate(dedup(markers))]
        leg2 = plt.legend(
            handles=h, 
            labels=sample_ratio_to_marker_map.keys() if markers_signify=="sampling_rate" else prune_strat_to_marker_map.keys(), 
            loc="lower left", 
            title="Sampling rate")
        ax.add_artist(leg2)

    # Legend 3: Num bucket / Size
    h = [
        plt.plot([],[], marker='o', ls="", color="grey", ms=size/25)[0]
        for size in sorted(set(sizes))]
    leg3 = plt.legend(
        handles=h, 
        labels=sorted(set(setting.get("num_histogram_buckets", 16) for setting in point_data)), 
        loc="center right", 
        title="Max Histogram Size")
    ax.add_artist(leg3)

    plt.show()


    return fig, ax