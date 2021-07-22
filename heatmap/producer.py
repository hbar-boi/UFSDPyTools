import matplotlib.pyplot as plt
import numpy as np
import ROOT as rt
from array import array

IN_PATH = "/media/work/Waveforms/run4/analysis/amplitudes/new/100-200-2.root"
CHANNELS = [7, 8]

START_X = 0
END_X = 1000
STEP_X = 10

START_Y = 0
END_Y = 1000
STEP_Y = 10

WITHIN_STD = 1

def init():
    data = [[[] for j in range(101)] for i in range(101)]

    file = rt.TFile.Open(IN_PATH, "READ")
    tree = file.Get("data")

    values = {}
    for channel in CHANNELS:
        value = array("d", [0.0])
        tree.SetBranchAddress("amp{}".format(channel), value)
        values[channel] = value

    position = rt.std.vector("double")()
    tree.SetBranchAddress("pos", position)

    print("Loading data...", end = "\n\n")

    entries = tree.GetEntries()
    for entry in range(entries):
        tree.GetEntry(entry)

        x, y = int(position[0] / 10), int(position[1] / 10)

        output = []
        for channel in CHANNELS:
            output.append(values[channel][0])

        data[x][y].append(output)
        if(entry % 100000 == 0):
            print("Progress: {}/{}".format(entry, entries))

    file.Close()

    print("\nApplying corrections...", end = "\n\n")

    data = np.asarray(data)
    height, width = data.shape
    draw = np.empty((height, width))
    for x in range(height):
        for y in range(width):
            # Remove extraneous peaks
            channels = np.asarray(data[x][y])

            samples = channels.transpose()
            means = [np.mean(channel) for channel in samples]
            stds = [np.std(channel) for channel in samples]

            entries, depth = channels.shape
            new = []
            for entry in range(entries):
                delete = False
                for chn in range(depth):
                    mean = means[chn]
                    std = stds[chn]

                    value = channels[entry][chn]
                    if np.abs(mean - value) > WITHIN_STD * std:
                        print("Removed {:.0f} vs {:.0f} at ({}, {})".format(
                            value, mean, x, y))
                        delete = True

                    if value < 80:
                        channels[entry][chn] = 0

                if not delete:
                    new.append(channels[entry])

            draw[x][y] = np.mean(np.asarray(new))

    def out(d, q):
        return (draw)[d, q]

    m = np.arange(0, 101)
    n = np.arange(0, 101)
    x, y = np.meshgrid(m, n)

    z = out(x, y)
    fig, ax = plt.subplots(1, figsize = (7, 7))
    ax.set_ylabel("um", fontsize = 10)
    ax.set_xlabel("um", fontsize = 10)
    cm = ax.pcolormesh(x, y, z, shading = "auto", cmap = "viridis")
    fig.colorbar(cm)
    plt.show()

if __name__ == "__main__":
    init()
