import re
import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import scipy.linalg
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from multiprocessing import Manager, Process
ps = PorterStemmer()
import shutil
import os

from time import time_ns
def get_time():
    return time_ns() / 1e9 / 60

def load_words(path):
    with open(path, encoding="UTF-8") as f:
        T = f.read()
        W = T.split("\n")
        return W

def load_docs(path):
    with open(path, encoding="UTF-8") as f:
        T = f.read()
        D = T.split("\n")
        return D

def load_A(path):
    return scipy.sparse.load_npz(path)

def load_all(path):
    W_path = path + "IDX_TO_WORD.txt"
    D_path = path + "IDX_TO_DOCS.txt"
    A_path = path + "A.npz"

    W = load_words(W_path)
    if W is None:
        raise IOError(f"Cannot open file {W_path}")

    D = load_docs(D_path)
    if D is None:
        raise IOError(f"Cannot open file {D_path}")

    A = load_A(A_path)
    if A is None:
        raise IOError(f"Cannot open file {A_path}")

    return A, W, D


def get_query_vector(q, W):
    q = word_tokenize(q)
    q = [ps.stem(w) for w in q]
    return np.array([1 if w in q else 0 for w in W]), q


def get_test_queries(N, D):
    Q, Links = [], []
    for d in D[:N]:
        with open(d, encoding="UTF-8") as f:
            Links.append(f.readline())
            title = f.readline()
            while title == "\n":
                title = f.readline()
            title = title.split("-")[0]
            Q.append(title)
    return Q, Links




if __name__ == '__main__':

    MOST_IMPORTANT = 25
    DOCS_TO_TESTS = 200
    t1 = get_time()

    path = "../parsed/5_120000_wikipedia1/"
    A, W, D = load_all(path)

    K = [int(k) for k in os.listdir(path + "svd") if int(k) % 2 == 0]
    K.sort()
    Res = dict([(k, 0) for k in K])
    Q, Links = get_test_queries(DOCS_TO_TESTS, D)
    for k in K:
        dirpath = path + "svd/" + str(k)

        dirpath += "/"
        U = np.load(dirpath + "U.npy")
        s = np.load(dirpath + "s.npy")
        Vt = np.load(dirpath + "Vt.npy")
        S = np.diag(s) @ Vt

        t = get_time()
        if not f"test{DOCS_TO_TESTS}.txt" in os.listdir(dirpath):
            for j, (q, l) in enumerate(zip(Q, Links)):
                print(f"k={k}: doc:{j}, q:{q} , link:{l},     {get_time() - t1} min")
                qv, _ = get_query_vector(q, W)

                cosfi_ = qv @ U @ S
                Div = np.linalg.norm(S, axis=0) * np.linalg.norm(qv)
                I = np.abs(cosfi_ / Div)

                Importance = [(v, i) for i, v in enumerate(I)]
                Importance.sort()
                N = len(Importance)
                MostImportant = set([i for _, i in Importance[N-MOST_IMPORTANT:]])
                if l in MostImportant:
                    Res[k] += 1

            with open(dirpath + f"test{DOCS_TO_TESTS}.txt", "w") as f:
                f.write(f"{Res[k]}\n{get_time() - t} min")
        else:
            with open(dirpath + f"test{DOCS_TO_TESTS}.txt") as f:
                Res[k] = float(f.readline())
    with open(path + f"svd_tests_{DOCS_TO_TESTS}.txt", "w") as f:
        f.write("\n".join([f"{k}:{Res[k]}" for k in Res.keys()]))

