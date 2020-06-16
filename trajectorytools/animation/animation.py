import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from . import plotter
from .scatter import Scatter


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""

    def __init__(self, datasets=[], plotters=[]):
        self.datasets = datasets
        self.plotters = plotters

    def prepare(
        self, interval=20, limits=[-1, 1, -1, 1], axis_off=True, fig_ax=None
    ):
        frames = self.datasets[0].shape[0] - 1
        self.scatters = [
            Scatter(self.datasets[i], plotter=self.plotters[i])
            for i in range(len(self.datasets))
        ]
        # Setup the figure and axes...
        if fig_ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig, self.ax = fig_ax

        self.fig.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=None, hspace=None
        )
        if axis_off:
            self.ax.axis("off")
        self.ax.axis(limits)
        self.ax.set_aspect("equal")
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update,
            interval=interval,
            frames=frames,
            init_func=self.setup_plot,
            blit=True,
            repeat=False,
        )

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        scat_tuple = ()
        for scatter in self.scatters:
            scat_tuple = scat_tuple + scatter.first_plot(self.ax)
        return scat_tuple

    def update(self, i):
        """Update the scatter plot."""
        scat_tuple = ()
        for scatter in self.scatters:
            scat_tuple = scat_tuple + scatter.update_plot()
        return scat_tuple

    def show(self):
        plt.show()

    def save(self, video_file_name, fps=10):
        mywriter = animation.FFMpegWriter(codec="h264", fps=fps)
        self.ani.save(video_file_name, writer=mywriter)

    def __add__(self, other):
        return AnimatedScatter(
            self.datasets + other.datasets, self.plotters + other.plotters
        )


def scatter(positions, **kwargs):
    plotters = [plotter.simple(**kwargs)]
    return AnimatedScatter([positions], plotters=plotters)


def scatter_circle(positions, **kwargs):
    plotters = [plotter.circle(**kwargs)]
    return AnimatedScatter([positions], plotters=plotters)


def scatter_labels(positions, **kwargs):
    plotters = [plotter.labels(**kwargs)]
    return AnimatedScatter([positions], plotters=plotters)


def scatter_ellipses(positions, velocities, **kwargs):
    data = np.concatenate((positions, velocities), axis=-1)
    plotters = [plotter.ellipse(**kwargs)]
    return AnimatedScatter([data], plotters=plotters)


def scatter_ellipses_color(positions, velocities, color, **kwargs):
    data = np.concatenate((positions, velocities, color), axis=-1)
    plotters = [plotter.ellipse(**kwargs)]
    return AnimatedScatter([data], plotters=plotters)


def scatter_vectors(positions, velocities, **kwargs):
    data = np.concatenate((positions, velocities), axis=-1)
    plotters = [plotter.vectors(**kwargs)]
    return AnimatedScatter([data], plotters=plotters)
