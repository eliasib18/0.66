import datetime as dt
import pathlib
import matplotlib
import matplotlib.pyplot as plt
file_path = pathlib.Path(__file__).parent.absolute()

class PlotValues():
    def __init__(self, logs={}):
        self.i = 0
        self.x = []
        self.values = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epochs, data):
        self.x.append(self.i)
        self.values.append(data)
        self.i += 1
        plt.cla()
        plt.plot(self.x, self.values, label="Reward")
        plt.xlabel('epochs')
        plt.ylabel('Reward')
        plt.legend()
        plt.show(block=False)
        plt.pause(5)

    def on_train_end(self, title):
        plt.show()
        today = dt.datetime.now().strftime("%Y-%m-%d")
        values_file = file_path/ f'figures/{title}_values_{today}.png'
        plt.savefig(values_file)