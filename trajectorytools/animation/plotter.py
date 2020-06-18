from collections import namedtuple

import matplotlib as mpl
import numpy as np

Plotter = namedtuple("Plotter", "first update")


def ellipse(color="c", size=[0.04, 0.02]):
    def plot_function(xy, ax):
        patches = tuple(
            mpl.patches.Ellipse(
                x[:2],
                size[0],
                size[1],
                np.degrees(np.arctan2(x[3], x[2])),
                color=color,
                animated=True,
            )
            for x in xy
        )
        for patch in patches:
            ax.add_patch(patch)
        return patches

    def update_function(xy, ax, patches, frame=None):
        for i, patch in enumerate(patches):
            patch.center = xy[i, :2]
            patch.angle = np.degrees(np.arctan2(xy[i, 3], xy[i, 2]))
            if len(xy[i, :]) > 4:  # If there is color information, use it
                patch.set_facecolor(xy[i, 4:])
                patch.set_edgecolor(xy[i, 4:])
            patch.stale = True
        return patches

    return Plotter(first=plot_function, update=update_function)


def simple(marker="o", color=[1, 0, 0]):
    def plot_function(xy, ax):
        return (
            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                facecolor="none",
                edgecolors=color,
                marker=marker,
                animated=True,
            ),
        )

    def update_function(xy, ax, scat, frame=None):
        scat[0].set_offsets(xy[:, :2])
        return scat

    return Plotter(first=plot_function, update=update_function)


def vectors(color="b", k=1):
    def plot_function(xy, ax):
        lines = tuple(
            ax.plot(
                [xy[i, 0], xy[i, 0] + xy[i, 2] * k],
                [xy[i, 1], xy[i, 1] + xy[i, 3] * k],
                color,
                animated=True,
            )[0]
            for i in range(xy.shape[0])
        )
        return lines

    def update_function(xy, ax, arrows, frame=None):
        for i in range(xy.shape[0]):
            arrows[i].set_xdata([xy[i, 0], xy[i, 0] + xy[i, 2] * k])
            arrows[i].set_ydata([xy[i, 1], xy[i, 1] + xy[i, 3] * k])
        return arrows

    return Plotter(first=plot_function, update=update_function)


def circle(color="k", radius=1):
    def plot_function(xy, ax):
        patches = tuple(
            mpl.patches.Circle((x[0], x[1]), radius, fill=False, animated=True)
            for x in xy
        )
        for patch in patches:
            ax.add_patch(patch)
        return patches

    def update_function(xy, ax, lines, frame=None):
        for i, patch in enumerate(lines):
            patch.center = xy[i, 0], xy[i, 1]
        return lines

    return Plotter(first=plot_function, update=update_function)


def labels(color="k", fontsize=15, labels=None, colors=None):
    def plot_function(xy, ax):
        if len(labels.shape) == 2:
            label_list = labels[0].tolist()
        else:
            label_list = labels.tolist()
        patches = tuple(ax.text(x[0], x[1], l) for l, x in zip(label_list, xy))
        if colors is not None:
            if isinstance(colors[0], list):
                color_list = colors[0]
            else:
                color_list = colors
            for patch, color in zip(patches, color_list):
                patch.set_color(color)
        return patches

    def update_function(xy, ax, lines, frame=None):

        for i, patch in enumerate(lines):
            patch.set_position((xy[i, 0], xy[i, 1]))
            patch.set_text(labels[frame][i])

        if colors is not None:
            if isinstance(colors[0], list):
                color_list = colors[frame]
                for patch, color in zip(lines, color_list):
                    patch.set_color(color)

        if len(labels.shape) == 2:
            labels[0].tolist()

        return lines

    return Plotter(first=plot_function, update=update_function)
