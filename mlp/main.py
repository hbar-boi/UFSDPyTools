from array import array
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import ROOT as rt
import numpy as np

IN_PATH = "/media/work/Waveforms/run4/analysis/amplitudes/new/100-200-2.root"
CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

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
            print("Progress: {}/{} events".format(entry, entries))

    file.Close()

    print("\nApplying corrections... ", end = "")

    highest = 0

    data = np.asarray(data)
    height, width = data.shape
    for x in range(height):
        for y in range(width):
            # Remove extraneous peaks
            channels = np.asarray(data[x][y])
            local = np.amax(channels)
            highest = local if local > highest else highest

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
                        delete = True
                        break

                if not delete:
                    new.append(channels[entry])

            # print("Done at ({}, {})".format(x, y))
            data[x][y] = new

    print("Done!", end = "")
    print("\nBuilding vectors... ", end = "")

    a = [] # Coordinates
    b = [] # Peaks

    for x in range(height):
        for y in range(width):
            entries = np.asarray(data[x][y])
            for entry in entries:
                add = False
                new = []
                for e in entry.tolist():
                    if e > 120:
                        new.append(e / highest)
                        add = True
                    else:
                        new.append(0)
                if add:
                    b.append(new)
                    a.append([x, y])

    print("Done!", end = "")
    print("\nStarting training... ", end = "")

    train(b, a)

def train(x, y):
    np.random.seed(1)
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.25,
        random_state = 42)
    regressor = MLPRegressor(hidden_layer_sizes = (7), random_state = 42,
        activation = "logistic", max_iter = 1000).fit(xTrain, yTrain)

    print("Done!\n")

    print("Test score: {:.2f}".format(regressor.score(xTest, yTest)))
    yPredicted = regressor.predict(xTest)

    resolution = np.absolute(np.asarray(yTest - yPredicted))
    best = (np.partition(-resolution, 5))[0:5]
    worst = (-np.partition(-resolution, 5))[:5]
    average = np.mean(resolution)

    print("Best resolution: {}".format(best))
    print("Average resolution: {}".format(average))
    print("Worst resolution: {}".format(worst))

if __name__ == "__main__":
    init()
