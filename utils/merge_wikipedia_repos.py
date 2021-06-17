import os
import re
import shutil


def foolish(S):
    return S[0] + ":" + S[1], int(S[2])

def notfoolish(S):
    return S[0], int(S[1])

def merge(path_to, path_from):
    with open(path_to + "idx.txt") as f:
        R_to = dict([notfoolish(l.split("|")) for l in f.read().split("\n")])
    with open(path_from + "idx.txt") as f:
        R_from = dict([notfoolish(l.split(":")) for l in f.read().split("\n")])

    new_idx = 0
    R_to_links = set(R_to.keys())
    R_to_indexes = set(R_to.values())
    max_idx = max(R_to_indexes)

    n = len(R_from)
    for i, link in enumerate(R_from.keys()):
        print(f"{i}/{n}")
        idx = R_from[link]
        if link not in R_to_links:
            if new_idx <= max_idx:
                while new_idx in R_to_indexes:
                    new_idx += 1
            else:
                new_idx += 1
            shutil.move(path_from + f"readable_texts/{idx}.txt", path_to + f"readable_texts/{new_idx}.txt")
            shutil.move(path_from + f"raw_html/{idx}.txt", path_to + f"raw_html/{new_idx}.txt")
            R_to[link] = new_idx
    with open(path_to + "idx.txt", "w") as f:
        f.write("\n".join([f"{l}|{R_to[l]}" for l in R_to.keys()]))

def repair_indexing(path):
    with open(path + "idx.txt", "r") as f:
        T = f.read().split("\n")
        T = [l.split("|") for l in T]
        for i in range(1, len(T)):
            T[i][1] = str(int(T[i-1][1]) + 1)
    with open(path + "idx.txt", "w") as fw:
        fw.write("\n".join([f"{l[0]}|{l[1]}" for l in T]))


if __name__ == '__main__':
    # path_to = "D:/STUDIA-s/sem4/MOWNiT/laby/lab8/wikipedia1/"
    # path_from = "D:/STUDIA-s/sem4/MOWNiT/laby/lab8/wikipedia/"

    # merge(path_to, path_from)
    pass