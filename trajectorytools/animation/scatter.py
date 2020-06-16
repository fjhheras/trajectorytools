class Scatter(object):
    def __init__(self, data, plotter=None):
        self.data = data
        self.stream = self.data_stream()
        self.plotter = plotter
        self.frame = None

    def first_plot(self, ax):
        self.frame = 0
        xy = next(self.stream)
        self.ax = ax
        self.scat = self.plotter.first(xy, ax)
        return self.scat

    def update_plot(self):
        self.frame += 1
        xy = next(self.stream)
        self.scat = self.plotter.update(
            xy, self.ax, self.scat, frame=self.frame
        )
        return self.scat

    def data_stream(self):
        while True:
            for i in range(self.data.shape[0]):
                yield self.data[i]
