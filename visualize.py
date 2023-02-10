import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def _pretty_dict_str(d):
    return "\n".join(f"{k}: {v}" for k, v in d.items())


def pause_for_visuals():
    """Use to pause program execution when using plt.show(block=False)"""
    plt.show()

def plot_errors(data: dict[str, list[float]], title_data: dict|str):
    fig = plt.figure(tight_layout=True, figsize=[12, 6])  # default figsize=[6.4, 4.8] ([width, height] in inches)
    gs = gridspec.GridSpec(3, len(data)//3 + ((len(data) % 3) != 0))

    for grid_idx, (name, arr) in enumerate(data.items()):
        grid_item = gs[grid_idx%3, grid_idx//3]
        ax = fig.add_subplot(grid_item)

        ax.hist(arr, range=(0, max(arr)), bins=50)
        ax.set_ylabel("Count")
        ax.set_xlabel(name)

    if isinstance(title_data, dict):
        fig.suptitle(_pretty_dict_str(title_data))
    else:
        fig.suptitle(title_data)
        
    fig.align_labels()
    plt.show()


# Hover code inspiration: https://stackoverflow.com/a/47166787/8132000
def scatterplot(x, y, point_data):
    fig, ax = plt.subplots(tight_layout=True, figsize=[12, 6])
    scp = ax.scatter(x, y, c=[setting["stats_type"].value for setting in point_data])

    ax.set_ylabel("Stats size")
    ax.set_xlabel("Mean error")

    annotation = ax.annotate("", xy=(0, 0), xytext=(10, -20), textcoords="offset points", bbox={"boxstyle": "round", "fc": "w"})
    annotation.set_visible(False)

    def update_annotation(hover_target_idx):
        target_pos = scp.get_offsets()[hover_target_idx]
        annotation.xy = target_pos
        annotation.set_text(_pretty_dict_str(point_data[hover_target_idx]))

    def on_hover(event):
        contains, targets = scp.contains(event)
        if contains:
            update_annotation(targets['ind'][0])
            annotation.set_visible(True)
            fig.canvas.draw_idle()
        else:
            annotation.set_visible(False)
            fig.canvas.draw_idle()


    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    plt.show()
