import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse

# Scene

__all__ = [
    "Fish",
    "Scene",
    "plot_individual_distribution",
    "plot_individual_distribution_of_vector",
]


class Fish:
    default_ellipse_params = {
        "width": 1,
        "height": 1 / 2,
        "fc": "b",
        "ec": "k",
        "linewidth": 0.03,
    }

    def __init__(
        self,
        xy,
        v,
        restricted=False,
        ellipse_params=None,
        color="k",
        marker_size=0.04,
        vel_factor=10,
        velocity_marker=True,
    ):
        self._xy = xy
        self._v = v
        self.vel_factor = vel_factor
        self.restricted = restricted

        if ellipse_params is None:
            ellipse_params = {}
        ellipse_params = {**self.default_ellipse_params, **ellipse_params}
        ellipse_params["angle"] = np.degrees(np.arctan2(v[1], v[0]))
        self.body = Ellipse(xy=xy, **ellipse_params)
        self.velocity_line = Line2D(
            [xy[0], self.xy_vel[0]],
            [xy[1], self.xy_vel[1]],
            color=color,
            linewidth=0.4,
        )
        self.artists = [self.body, self.velocity_line]
        if velocity_marker:
            self.velocity_marker = Circle(
                xy=self.xy_vel, radius=marker_size / 10, fc="k"
            )
            self.artists += [self.velocity_marker]

    @property
    def xy_vel(self):
        return self.position + self.velocity * self.vel_factor

    @property
    def figure(self):
        return self.body.figure

    @property
    def position(self):
        return self._xy  # self.body.center

    @position.setter
    def position(self, xy):
        if not self.restricted:
            self._xy = xy
            self.body.center = xy
            self.velocity_marker.center = self.xy_vel
            self.velocity_line.set_data(
                [xy[0], self.xy_vel[0]], [xy[1], self.xy_vel[1]]
            )
            self.body.stale = True

    @property
    def velocity(self):
        return self._v  # self.body.center

    @velocity.setter
    def velocity(self, v):
        if self.restricted:
            v = v.copy()
            v[1] = max(0, v[1])
            v[0] = 0.0
        self._v = v
        self.body.angle = np.degrees(np.arctan2(v[1], v[0]))
        self.velocity_marker.center = self.position + v * self.vel_factor
        self.velocity_line.set_data(
            [self.position[0], self.xy_vel[0]],
            [self.position[1], self.xy_vel[1]],
        )
        self.body.stale = True

    def add_to_axis(self, ax):
        for artist in self.artists:
            ax.add_artist(artist)
            artist.set_clip_box(ax.bbox)
        self.velocity_line.axes = ax
        self.axes = ax  # self.body.axes


class Scene:
    def __init__(
        self, fish, ax, target=None, focal_acceleration=False, vel_factor=0.3
    ):
        for f in fish:
            fish_artist = Fish(
                f["position"],
                f["velocity"],
                ellipse_params=f["ellipse"],
                vel_factor=vel_factor,
            )
            fish_artist.add_to_axis(ax)
        if target is not None:
            ax.add_artist(plt.Circle((target[0], target[1]), 0.05, color="k"))
        if focal_acceleration is not None:
            ax.plot(
                [0, focal_acceleration[0]],
                [0, focal_acceleration[1]],
                "k",
                linewidth=0.03,
            )

    @classmethod
    def from_frame(
        cls,
        frame,
        ax,
        color=None,
        edgecolor=None,
        body_length=1,
        target=None,
        focal_acceleration=True,
        vel_factor=0.3,
    ):
        assert len(frame.shape) == 2
        fish = []
        for i in range(frame.shape[0]):
            new_fish = {
                "position": np.array([frame[i, 0], frame[i, 1]]),
                "velocity": np.array([frame[i, 2], frame[i, 3]]),
            }
            if color is not None and color[i] is not None:
                new_fish["ellipse"] = {
                    "fc": color[i],
                    "width": body_length,
                    "height": 0.5 * body_length,
                }

            fish.append(new_fish)
        if focal_acceleration:
            focal_acceleration = frame[0, 4:]  # *DELTA_A*0.3
        else:
            focal_acceleration = None

        return cls(
            fish,
            ax,
            target=target,
            vel_factor=vel_factor,
            focal_acceleration=focal_acceleration,
        )


### Histogram / diagram stuff below


def get_spaced_colors(n, cmap="jet"):
    RGB_tuples = matplotlib.cm.get_cmap(cmap)
    return [RGB_tuples(i / n) for i in range(n)]


def subplots_row_and_colums(number_of_individuals):
    number_of_columns = int(np.sqrt(number_of_individuals))
    number_of_rows = int(np.ceil(number_of_individuals / number_of_columns))
    return number_of_rows, number_of_columns


def no_ticks(ax):
    ax.set_yticklabels([]), ax.set_yticks([])
    ax.set_xticklabels([]), ax.set_xticks([])


def with_ordering(old_func):
    def new_func(scalar_or_vector, order_by=None, indices=None, **kwargs):
        number_of_individuals = scalar_or_vector.shape[1]
        if order_by is None:
            if indices is None:
                indices = range(number_of_individuals)
        else:
            if indices is None:
                indices = np.argsort(order_by)
            else:
                print(
                    "Warning! Giving both indices and order_by. Taking indices"
                )
        return old_func(scalar_or_vector, indices, **kwargs)

    return new_func


@with_ordering
def plot_individual_distribution(variable, indices, nbins=25, ticks=False):
    number_of_individuals = len(indices)
    number_of_rows, number_of_columns = subplots_row_and_colums(
        number_of_individuals
    )
    fig, ax_arr = plt.subplots(
        number_of_rows, number_of_columns, sharex=True, sharey=True
    )
    min_variable = np.min(variable)
    max_variable = np.percentile(variable, 99)
    bins = np.linspace(min_variable, max_variable, nbins)
    for i, identity in enumerate(indices):
        ax = ax_arr[int(i / number_of_columns), i % number_of_columns]
        n, bins_edges = np.histogram(variable[:, identity], bins=bins)
        ax.plot(bins_edges[:-1] + np.diff(bins_edges)[0], n)
        if ticks is False:
            no_ticks(ax)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    return fig


@with_ordering
def plot_individual_distribution_of_vector(
    vector, indices, nbins=10, ticks=False
):
    if np.isnan(vector).any():
        print("Removing NaNs from data")
    number_of_individuals = len(indices)
    number_of_rows, number_of_columns = subplots_row_and_colums(
        number_of_individuals
    )
    fig, ax_arr = plt.subplots(number_of_rows, number_of_columns)
    v_max = 0
    min_x, max_x = (
        np.percentile(vector[:, :, 0][~np.isnan(vector[:, :, 0])], 1),
        np.percentile(vector[:, :, 0][~np.isnan(vector[:, :, 0])], 99),
    )
    min_y, max_y = (
        np.percentile(vector[:, :, 1][~np.isnan(vector[:, :, 1])], 1),
        np.percentile(vector[:, :, 1][~np.isnan(vector[:, :, 1])], 99),
    )
    print("X from {} to {}".format(min_x, max_x))
    print("Y from {} to {}".format(min_y, max_y))
    binsX = np.linspace(min_x, max_x, nbins)
    binsY = np.linspace(min_y, max_y, nbins)
    ax = []
    H = []
    for i, identity in enumerate(indices):
        ax.append(ax_arr[int(i / number_of_columns), i % number_of_columns])
        H.append(
            np.histogram2d(
                vector[:, identity, 0][
                    ~np.isnan(vector[:, identity, 0])
                ].flatten(),
                vector[:, identity, 1][
                    ~np.isnan(vector[:, identity, 1])
                ].flatten(),
                bins=(binsX, binsY),
            )[0]
        )
        v_max = max(v_max, H[i].max())
    for i, H_i in enumerate(H):
        ax[i].imshow(H_i, vmin=0, vmax=v_max, cmap="jet")
        ax[i].set_title(str(indices[i] + 1), fontsize=8)
        print(ax[i].get_xlim(), ax[i].get_ylim())
        # ax[i].set_title(str(indices[i]))
        if ticks is False:
            no_ticks(ax[i])
    return fig
