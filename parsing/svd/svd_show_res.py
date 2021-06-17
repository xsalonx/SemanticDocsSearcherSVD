
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    path = "../parsed/5_120000_wikipedia1/svd"
    K = os.listdir(path)
    R = {}
    for k in K:
        if "test200.txt" in os.listdir(path + f"/{k}"):
            with open(path + f"/{k}/test200.txt") as f:
                R[int(k)] = int(f.readline())

    X = list(R.keys())
    Y = [R[k] for k in X]
    plt.scatter(X, Y)
    plt.xlabel("SVD degree - k")
    plt.ylabel("number of listed titles")
    plt.title("Plot showing how many of 200 titles passed as query\n are listed in first twenty positions")
    plt.show()

