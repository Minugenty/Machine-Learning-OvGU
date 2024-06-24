import numpy as np
import argparse


def mat(dataset):
    data = np.genfromtxt(dataset, delimiter=',')
    x = data[:,0:-1].reshape(-1, data.shape[1]-1)
    ones = np.ones([x.shape[0],1])
    x = np.concatenate([ones, x],1)
    y = data[:,-1].reshape(-1,1)
    w = np.zeros([1, x.shape[1]])

    return x, y, w


def sse(x, y, w):
    e = np.power(((x@w.T)-y), 2)
    return np.sum(e)


def regressor(dataset,eta,t):
    x, y, w = mat(dataset)
    costmat,res=[],[]
    key=0
    while True:
        temp=[]
        temp.append(key)
        key += 1
        cost = sse(x,y,w)
        for ol in w:
            for il in ol:
                temp.append(il)
        temp.append(cost)
        costmat.append(cost)
        res.append(temp)
        w = w-(eta)*np.sum((x@w.T-y)*x,axis=0)

        if key>1:
            if (costmat[-2]-costmat[-1]) <= t:
                break
    for i in res:
        print(*i, sep=",")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='dataset.csv')
    parser.add_argument('--eta', help='learning rate')
    parser.add_argument('--threshold', help='threshold value')

    args = parser.parse_args()

    dataset = args.data
    eta = float(args.eta)
    threshold = float(args.threshold)
    regressor(dataset, eta, threshold)
