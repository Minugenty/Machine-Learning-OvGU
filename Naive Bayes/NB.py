import argparse
import csv
import pandas as pd
import numpy as np

def Prob(x, mean, variance):
    x = float(x)
    return (1/(np.sqrt(2 * 3.14 * variance))) * np.exp(-((x-mean)**2 / (2*variance)))

def Separate(column, label):
    Adata = []
    Bdata = []
    for i in range(len(label)):
        if label[i] == 'A':
            Adata.append(column[i])
        else:
            Bdata.append(column[i])
    Adata = list(map(float, Adata))
    Bdata = list(map(float, Bdata))
    return Adata, Bdata


def Naive_Bayes(data):
    c1_Adata, c1_Bdata = Separate(data[1], data[0])
    c2_Adata, c2_Bdata = Separate(data[2], data[0])
    p1_A = len(c1_Adata)/len(data)
    p1_B = len(c1_Bdata)/len(data)
    p2_A = len(c2_Adata)/len(data)
    p2_B = len(c2_Bdata)/len(data)
    meanA1 = np.mean(c1_Adata)
    meanA2 = np.mean(c2_Adata)
    meanB1 = np.mean(c1_Bdata)
    meanB2 = np.mean(c2_Bdata)
    varianceA1 = sum([(x-meanA1)**2 for x in c1_Adata]) / float(len(c1_Adata)-1)
    varianceB1 = sum([(x-meanB1)**2 for x in c1_Bdata]) / float(len(c1_Bdata)-1)
    varianceA2 = sum([(x-meanA2)**2 for x in c2_Adata]) / float(len(c2_Adata)-1)
    varianceB2 = sum([(x-meanB2)**2 for x in c2_Bdata]) / float(len(c2_Bdata)-1)
    count = 0
    for i in range(len(data)):
        gA1 = Prob(data.iloc[i, 1], meanA1, varianceA1)
        gB1 = Prob(data.iloc[i, 1], meanB1, varianceB1)
        gA2 = Prob(data.iloc[i, 2], meanA2, varianceA2)
        gB2 = Prob(data.iloc[i, 2], meanB2, varianceB2)

        pA = p1_A * gA1 * gA2
        pB = p1_B * gB1 * gB2
        if pA > pB:
            label = "A"
        elif pB > pA:
            label = "B"
        else:
            pass
        if data.iloc[i, 0] != label:
            count += 1

    print(meanA1, varianceA1, meanA2, varianceA2, p1_A, sep=",")
    print(meanB1, varianceB1, meanB2, varianceB2, p1_B, sep=",")
    print(count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Your CSV data file")

    args = parser.parse_args()
    file = args.data

    with open(file) as File:
        data = csv.reader(File, delimiter=',')
        data = pd.DataFrame(data)

    Naive_Bayes(data)
