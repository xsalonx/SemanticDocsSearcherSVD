
import scipy.sparse
import scipy.sparse.linalg
import os
import numpy as np
from multiprocessing import Manager, Process

def make_svd_(dirpath, R, valI):
    A = scipy.sparse.load_npz(dirpath + "A.npz")

    for k in R:
        print(f"({valI.value}, {k}) ")
        svd_dirpath = dirpath + "svd/" + str(k)
        if str(k) not in os.listdir(dirpath + "svd"):
            U, s, Vt = scipy.sparse.linalg.svds(A, k=k, which='LM')

            os.mkdir(svd_dirpath)
            svd_dirpath += "/"
            np.save(svd_dirpath + "U", U)
            np.save(svd_dirpath + "s", s)
            np.save(svd_dirpath + "Vt", Vt)
            print(f"Done {k}")
        else:
            print(f"{k} already calculated")
        valI.value += 1



def make_svds_multiprocess(N, low, step, dirpath, processes_numb=2):
    R = list(low + i * step for i in range(N))
    h = N // processes_numb
    multi_R = [R[i * h: (i + 1) * h if i < processes_numb - 1 else N] for i in
               range(processes_numb)]
    Processes = []
    valI = Manager().Value("i", 0)
    for R in multi_R:
        Processes.append(Process(target=make_svd_, args=(dirpath, R, valI)))
    for p in Processes:
        p.start()
    for p in Processes:
        p.join()

class A:
    def __init__(self):
        self.value = 0

def make_svds(N, low, step, dirpath):
    R = list(low + i * step for i in range(N))

    make_svd_(dirpath, R, A())

if __name__ == '__main__':
    parsed_files_path = "../parsed/5_120000_wikipedia1/"

    print("\nSVDs calculating for k=...")
    # make_svds(40, 150, 5, dirpath)
    K = [1050, 1200]
    make_svd_(parsed_files_path, K, A())