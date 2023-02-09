import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_errors(data: dict[str, list[float]], override_settings: dict):
    fig = plt.figure(tight_layout=True, figsize=[12, 6])  # default figsize=[6.4, 4.8] ([width, height] in inches)
    gs = gridspec.GridSpec(3, len(data)//3 + ((len(data) % 3) != 0))

    for grid_idx, (name, arr) in enumerate(data.items()):
        grid_item = gs[grid_idx%3, grid_idx//3]
        ax = fig.add_subplot(grid_item)

        ax.hist(arr, range=(0, max(arr)), bins=50)
        ax.set_ylabel("Count")
        ax.set_xlabel(name)

    fig.suptitle("\n".join(f"{k}: {v}" for k, v in override_settings.items()))
    fig.align_labels()
    plt.show()

