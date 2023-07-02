from math import log2
from typing import Literal
from matplotlib import cbook
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.markers as plt_markers
from matplotlib.text import Text

from compute_structures import PruneStrat
from settings import settings

def _pretty_dict_str(d):
    def fix_enum_list(l):
        return [e.name for e in l]

    return "\n".join(
        f"{k}: {v}" if k != "prune_strats" else f"{k}: {fix_enum_list(v)}"
        for k, v in d.items()
    )


def pause_for_visuals():
    """Use to pause program execution when using plt.show(block=False)"""
    plt.show()

def plot_errors(data: dict[str, list[float]], title_data: dict|str):
    fig = plt.figure(tight_layout=True, figsize=[12, 6])  # default figsize=[6.4, 4.8] ([width, height] in inches)
    gs = gridspec.GridSpec(3, len(data)//3 + ((len(data) % 3) != 0))

    for grid_idx, (name, arr) in enumerate(data.items()):
        grid_item = gs[grid_idx%3, grid_idx//3]
        ax = fig.add_subplot(grid_item)

        ax.hist(arr, range=(0, max(arr)), bins=100)
        ax.set_ylabel("Count")
        ax.set_xlabel(name)
        ax.set_yscale('log')

    if isinstance(title_data, dict):
        fig.suptitle(_pretty_dict_str(title_data))
    else:
        fig.suptitle(title_data)
        
    fig.align_labels()
    plt.show()


# Hover code inspiration: https://stackoverflow.com/a/47166787/8132000
# Marker code inspiration: https://stackoverflow.com/a/52303895/8132000
def scatterplot(x, y, point_data, ylabel="", xlabel="", filename=None):
    markers_signify: Literal["sampling_rate", "prune_strats"] | None = None

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
        # if all(len(pd["prune_strats"]) <= 1 for pd in point_data):
        #     # for pd in point_data:
        #     #     pd["prune_strats"] = pd["prune_strats"][0] if pd["prune_strats"] else "No Prune Strat"
        # else:
        
        for pd in point_data:
            pd["prune_strats"] = tuple(pd["prune_strats"])

        prune_strat_to_marker_map = dict(zip(set(p["prune_strats"] for p in point_data), allowed_markers))
        markers = [prune_strat_to_marker_map[p["prune_strats"]] for p in point_data]
    else:
        markers = []


    # colors = [PruneStrat.UNIQUE_SUFFIX in setting["prune_strats"] for setting in point_data]
    # colors = [PruneStrat.NO_TYPED_INNER_NODES in setting["prune_strats"] for setting in point_data]

    def add_markers(scp):
        """
        matplotlib doesn't support a list of markers like it does for colors,
        so we have to set markers manually if we want more than one.
        """
        if not markers: return

        paths = [
            marker_obj.get_path().transformed(marker_obj.get_transform())
            for marker_obj in map(plt_markers.MarkerStyle, markers)
        ]
        scp.set_paths(paths)

    def get_true_annotation_extent(annotation):
        """
        AFAICT, all annotation and Text(annotation) methods that calculate extent
        return a unit bounding box (0, 0, 1, 1) whenever the annotation
        is out of bounds in any way. This makes them absolutely useless 
        for my case. 

        This function should always return the size of the annotation 
        if it were to be drawn, no matter if it actually can be drawn or not. 
        """

        # Source: matplotlib(==3.6.3)/text.py:914-919
        with cbook._setattr_cm(annotation.figure, dpi=fig.dpi):
            bbox, info, descent = annotation._get_layout(annotation._renderer)
            x, y = annotation.get_unitless_position()
            x, y = annotation.get_transform().transform((x, y))
            bbox = bbox.translated(x, y)
            return bbox

    def update_annotation(hover_target_idx, cursor_x, cursor_y):
        """Update annotation box content with the stats of the hover target"""

        # Update annotation text
        target_pos = scp.get_offsets()[hover_target_idx]
        # Coordinate system used by default is "offset points"
        annotation.xy = target_pos
        annotation.set_text(_pretty_dict_str(point_data[hover_target_idx]))
        
        # Update annotation position to avoid it going out of bounds
        coord_transformer = ax.transData.inverted()  # Inverted (data => pixel) transformer
        canv_width, canv_height = fig.canvas.get_width_height(physical=True)
        anno_bbox = get_true_annotation_extent(annotation)
        anno_width = anno_bbox.x1 - anno_bbox.x0
        anno_height = anno_bbox.y1 - anno_bbox.y0

        shift_left = cursor_x > canv_width//2
        shift_down = cursor_y > canv_height//2

        shifted_position = (
            cursor_x - anno_width if shift_left else cursor_x, 
            cursor_y - anno_height if shift_down else cursor_y
        )
        shifted_offset = (
            -10 if shift_left else 10,
            0 if shift_down else -20
        )

        annotation.xy = coord_transformer.transform(shifted_position)
        annotation.xyann = shifted_offset


    _prev_hover_target = None
    def on_hover(event):
        """On hover event handler for the plot. Displays an annotation when hovering over plot elements"""
        nonlocal _prev_hover_target
        contains, targets = scp.contains(event)
        if contains:
            target = targets['ind'][0]
            if target != _prev_hover_target:
                update_annotation(target, event.x, event.y)
                annotation.set_visible(True)
                fig.canvas.draw_idle()
            _prev_hover_target = target
        else:
            _prev_hover_target = None
            annotation.set_visible(False)
            fig.canvas.draw_idle()


    # fig, ax = plt.subplots(tight_layout=True, figsize=[10, 5])
    fig, ax = plt.subplots(tight_layout=True, figsize=[6, 4])
    # scp = ax.scatter(x, y, c=colors)
    scp = ax.scatter(x, y, c=colors, s=sizes)
    add_markers(scp)
    ax.set_ylabel(ylabel)#, fontsize=25)
    ax.set_xlabel(xlabel)#, fontsize=25)

    ax.set(ylim=(1, 200_001))
    ax.set(xlim=(0.1, 2.02))

    ## ANNOTATIONS ##
    annotation = ax.annotate("", xy=(0, 0), xytext=(10, -20), textcoords="offset points", bbox={"boxstyle": "round", "fc": "w"})
    annotation.set_visible(False)
    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    

    ## LEGENDS ##
    # Thanks for helping: https://stackoverflow.com/a/43814479
    leg1 = plt.legend(
        handles=scp.legend_elements()[0], 
        labels={setting["stats_type"].name: 0 for setting in point_data}.keys(),
        loc="upper right", 
        title="Statistics Type")
    ax.add_artist(leg1)


    if markers:
        h = [
            plt.plot([],[], marker=marker, ls="None", color="grey")[0]
            for i, marker in enumerate(set(markers))]
        leg2 = plt.legend(
            handles=h, 
            labels=sample_ratio_to_marker_map.keys() if markers_signify=="sampling_rate" else prune_strat_to_marker_map.keys(), 
            loc="lower left", 
            title="Sampling rate")
        ax.add_artist(leg2)

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


    if filename:
        full_fname = "viz_" + settings.stats.filename + "_" + filename + ".pdf"
        fig.savefig("figures/" + full_fname, format="pdf")