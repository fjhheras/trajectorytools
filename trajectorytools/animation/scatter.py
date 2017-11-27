class Scatter(object):
    def __init__(self, data, plotter = None):
        self.data = data
        self.stream = self.data_stream()
        self.plotter = plotter
    def first_plot(self, ax):
        xy = next(self.stream)
        self.ax = ax
        self.scat = self.plotter.first(xy, ax)
        return self.scat
    def update_plot(self):
        xy = next(self.stream)
        self.scat = self.plotter.update(xy, self.ax, self.scat)
        return self.scat
    def data_stream(self):
        while(True):
            for i in range(self.data.shape[0]):
                yield self.data[i]


