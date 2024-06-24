import argparse
import csv
import pandas as pd
import numpy as np


def neuralnets(data, eta, iterations):
    w_bias_h1 = 0.2
    w_a_h1 = -0.3
    w_b_h1 = 0.4
    w_bias_h2 = -0.5
    w_a_h2 = -0.1
    w_b_h2 = -0.4
    w_bias_h3 = 0.3
    w_a_h3 = 0.2
    w_b_h3 = 0.1
    w_bias_o = -0.1
    w_h1_o = 0.1
    w_h2_o = 0.3
    w_h3_o = -0.4
    for i in range(11):
        print('-', end=",")
    print(w_bias_h1, end=",")
    print(w_a_h1, end=",")
    print(w_b_h1, end=",")
    print(w_bias_h2, end=",")
    print(w_a_h2, end=",")
    print(w_b_h2, end=",")
    print(w_bias_h3, end=",")
    print(w_a_h3, end=",")
    print(w_b_h3, end=",")
    print(w_bias_o, end=",")
    print(w_h1_o, end=",")
    print(w_h2_o, end=",")
    print(w_h3_o)

    data[0] = data[0].astype(float)
    data[1] = data[1].astype(float)
    data[2] = data[2].astype(int)

    for i in range(iterations):
        for j in range(len(data)):
            v = w_a_h1 * data.iloc[j, 0] + w_b_h1 * data.iloc[j, 1] + w_bias_h1
            h1 = 1 /(1 + np.exp(-v))
            v = w_a_h2 * data.iloc[j, 0] + w_b_h2 * data.iloc[j, 1] + w_bias_h2
            h2 = 1 /(1 + np.exp(-v))
            v = w_a_h3 * data.iloc[j, 0] + w_b_h3 * data.iloc[j, 1] + w_bias_h3
            h3 = 1 /(1 + np.exp(-v))
            v = h1 * w_h1_o + h2 * w_h2_o + h3 * w_h3_o + w_bias_o
            out = 1 /(1 + np.exp(-v))

            error = data.iloc[j, 2] - out
            delta_o = out * (1 - out) * error
            delta_h1 = h1 * (1 - h1) * delta_o * w_h1_o
            delta_h2 = h2 * (1 - h2) * delta_o * w_h2_o
            delta_h3 = h3 * (1 - h3) * delta_o * w_h3_o

            w_h1_o = w_h1_o + eta * delta_o * h1
            w_h2_o = w_h2_o + eta * delta_o * h2
            w_h3_o = w_h3_o + eta * delta_o * h3
            w_bias_o = w_bias_o + eta * delta_o

            w_a_h1 = w_a_h1 + eta * delta_h1 * data.iloc[j, 0]
            w_b_h1 = w_b_h1 + eta * delta_h1 * data.iloc[j, 1]
            w_bias_h1 = w_bias_h1 + eta * delta_h1

            w_a_h2 = w_a_h2 + eta * delta_h2 * data.iloc[j, 0]
            w_b_h2 = w_b_h2 + eta * delta_h2 * data.iloc[j, 1]
            w_bias_h2 = w_bias_h2 + eta * delta_h2

            w_a_h3 = w_a_h3 + eta * delta_h3 * data.iloc[j, 0]
            w_b_h3 = w_b_h3 + eta * delta_h3 * data.iloc[j, 1]
            w_bias_h3 = w_bias_h3 + eta * delta_h3

            print(data.iloc[j,0], data.iloc[j,1], h1, h2, h3, out, data.iloc[j,2], delta_h1, delta_h2, delta_h3, delta_o, sep=",", end=",")
            print(w_bias_h1, end=",")
            print(w_a_h1, end=",")
            print(w_b_h1, end=",")
            print(w_bias_h2, end=",")
            print(w_a_h2, end=",")
            print(w_b_h2, end=",")
            print(w_bias_h3, end=",")
            print(w_a_h3, end=",")
            print(w_b_h3, end=",")
            print(w_bias_o, end=",")
            print(w_h1_o, end=",")
            print(w_h2_o, end=",")
            print(w_h3_o)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Your CSV data file")
    parser.add_argument("-l", "--eta", help="Learning Rate")
    parser.add_argument("-t", "--iterations", help="Iterations")

    args = parser.parse_args()
    file = args.data
    eta = float(args.eta)
    iterations = int(args.iterations)

    with open(file) as File:
        data = csv.reader(File, delimiter=',')
        data = pd.DataFrame(data)

    neuralnets(data, eta, iterations)
