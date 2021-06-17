import re
import os
from collections import defaultdict

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import os
import shutil

from multiprocessing import Process, Manager

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

ps = PorterStemmer()

from time import time_ns

def get_time():
    return time_ns() / 1e9 / 60


most_common_words = set(stopwords.words('english'))
with open("most_common", encoding="UTF-8") as f:
    t = f.read()
    sw = t.split("\n")
    for w in sw:
        most_common_words.add(w)


def get_time():
    return time_ns() / 1e9 / 60


def get_words_union_and_occurrances(pre_A):
    W = defaultdict(lambda: [0, 0], [])
    for d in pre_A:
        for w in pre_A[d].keys():
            W[w][0] += 1
            W[w][1] += pre_A[d][w]
    return W


def get_word_IDF(wf, N):
    return np.log(N) / wf


def wrapper(paths, TITLE_WEIGHT, pre_A, Titles_words, t1, process_index, h, valI):
    for i, path in enumerate(paths):
        print(f"proc:{process_index}, done {valI.value}, file:{i + process_index * h} - ", end="")

        D = defaultdict(lambda: 0, [])
        with open(path, encoding="UTF-8") as f:
            f.readline()
            # getting title of article
            title = f.readline()
            while title == "\n":
                title = f.readline()
            title = title.split("-")[0]
            WTT = word_tokenize(title)
            T = [ps.stem(w) for w in WTT if len(WTT) <= 2 or w not in most_common_words]

            text = f.read()
            text = re.sub("[ \n,~.:;\"\\\/<>{}()0-9\-_+|!=?@#$%^&*\[\]]+", " ", text)
            text = re.sub("[\n ]+", " ", text)

            words = word_tokenize(text)

            words = [ps.stem(w) for w in words if
                     len(w) > 2 and w not in most_common_words and 33 <= max([0] + [ord(l) for l in w]) <= 127]
            for w in words:
                D[w] += 1
            for w in T:
                D[w] += TITLE_WEIGHT

            pre_A[path] = dict([(w, D[w]) for w in D.keys()])
            Titles_words += T
            # with open("D:/STUDIA-s/sem4/MOWNiT/laby/lab8/wikipedia1/words_counts/1/" + path.split("/")[-1], "w", encoding="UTF-8") as fwc:
            #     fwc.write(f"TITLE_WEIGHT={TITLE_WEIGHT}\n#\n")
            #     fwc.write("\n".join([f"{w}:{D[w]}" for w in D.keys()]))

        valI.value += 1
        print(f"{get_time() - t1} min ")


def get_term_by_docs_matrix(Paths, COND_REMOVE_RATE=0.3, ABSOL_REMOVE_RATE=0.85, TITLE_WEIGHT=1, USE_TITLES=True,
                            processes_numb=2):
    """
    Analyzing text files with articles content using multiprocessing
    """

    manager = Manager()
    t1 = get_time()
    N = len(Paths)


    # Partitioning set of files for processes
    h = N // processes_numb
    proc_paths = [Paths[i * h: (i + 1) * h if i < processes_numb - 1 else N] for i in
                  range(processes_numb)]
    Processes = []

    # pre_A is a dict with keys - files paths and values - dicts containing calculated occurrences of words
    pre_A = dict()
    Titles_words = set()

    pre_A_proc_res = []
    Titles_words_proc_res = []
    valI = manager.Value('i', 0)
    for pi, paths in enumerate(proc_paths):
        pre_A__ = manager.dict()
        pre_A_proc_res.append(pre_A__)
        Titles_words__ = manager.list()
        Titles_words_proc_res.append(Titles_words__)

        Processes.append(Process(target=wrapper, args=(paths, TITLE_WEIGHT, pre_A__, Titles_words__, t1, pi, h, valI)))

    for p in Processes:
        p.start()
    for p in Processes:
        p.join()


    for pre_A__, Titles_words__ in zip(pre_A_proc_res, Titles_words_proc_res):
        for p in pre_A__.keys():
            pre_A[p] = pre_A__[p]
        Titles_words |= set(Titles_words__)

    """
    Constructing term-by-docs sparse matrix 
    """

    # W_occur - dict:: "word": [(number of file which contains "word"), (sum of occurrences "word" in each of all files)]
    W_occur = get_words_union_and_occurrances(pre_A)

    # Removing not necessary words
    for w in [w for w in W_occur.keys() if
              (W_occur[w][0] == W_occur[w][1] and W_occur[w][1] < 4) or
              W_occur[w][0] >= N * ABSOL_REMOVE_RATE or
              (W_occur[w][0] >= N * COND_REMOVE_RATE and (not USE_TITLES or w not in Titles_words))
              ]:
        W_occur.pop(w)

    IDX_TO_WORD = [w for w in W_occur.keys()]
    WORDS_SET = set(IDX_TO_WORD)
    WORD_TO_IDX = dict([(w, i) for i, w in enumerate(IDX_TO_WORD)])
    IDX_TO_DOC = [p for p in Paths]

    A = scipy.sparse.lil_matrix((len(IDX_TO_WORD), len(IDX_TO_DOC)), dtype=float)

    t2 = get_time()
    for i, p in enumerate(IDX_TO_DOC):
        print(f"(DOC {i}) ", end="")
        for w in pre_A[p].keys():
            if w in WORDS_SET:
                A[WORD_TO_IDX[w], i] = pre_A[p][w] / get_word_IDF(W_occur[w][0], N)

        # popping in order to free some memory space
        pre_A.pop(p)
        print(f"{get_time() - t1} min ")

    t3 = get_time()
    return scipy.sparse.csc_matrix(A), IDX_TO_WORD, W_occur, IDX_TO_DOC, (t2 - t1), (t3 - t2)


def main(N_docs, rootdirpath, wikipedia_repo, COND_REMOVE_RATE=0.3, ABSOL_REMOVE_RATE=0.85, TITLE_WEIGHT=1, USE_TITLES=True,
         processes_numb=2):
    # idxPath = wikipedia_repo + "idx.txt"
    # rawPath = wikipedia_repo + "raw_html/"
    parsedPath = wikipedia_repo + "readable_texts/"

    dirName = 1 + max([0] + [int(d.split("_")[0]) for d in os.listdir(rootdirpath)])
    dirName = f"{dirName}_{N_docs}_{wikipedia_repo[:-1].split('/')[-1]}"
    os.mkdir(rootdirpath + dirName)
    dirpath = rootdirpath + dirName + "/"
    os.mkdir(dirpath + "svd")
    Paths = [parsedPath + f"{n}.txt" for n in range(N_docs)]
    try:
        A, IDX_TO_WORD, W_occur, IDX_TO_DOC, t1, t2 = get_term_by_docs_matrix(Paths, COND_REMOVE_RATE=COND_REMOVE_RATE,
                                                                              ABSOL_REMOVE_RATE=ABSOL_REMOVE_RATE,
                                                                              TITLE_WEIGHT=TITLE_WEIGHT,
                                                                              USE_TITLES=USE_TITLES,
                                                                              processes_numb=processes_numb)

        scipy.sparse.save_npz(dirpath + "A", A)
        with open(dirpath + "IDX_TO_WORD.txt", "w", encoding="UTF-8") as f:
            f.write("\n".join(IDX_TO_WORD))
        with open(dirpath + "W_occur.txt", "w", encoding="UTF-8") as f:
            f.write("\n".join([f"{w}:{W_occur[w][0]}:{W_occur[w][1]}" for w in W_occur.keys()]))
        with open(dirpath + "IDX_TO_DOCS.txt", "w", encoding="UTF-8") as f:
            f.write("\n".join(IDX_TO_DOC))

        with open(dirpath + "metadata.txt", "w") as f:
            f.write(f"N_docs:{N_docs}\n")
            f.write(f"wikipedia_repo:{wikipedia_repo}\n")
            f.write(f"files parsing time:{t1} min\n")
            f.write(f"A creating time:{t2} min\n")
            f.write(f"COMMON_RATE:{COND_REMOVE_RATE}\n")


    except Exception as e:
        print(e)
        shutil.rmtree(dirpath[:-1], ignore_errors=True)

    return dirpath


if __name__ == '__main__':
    COND_REMOVE_RATE = 0.3
    ABSOL_REMOVE_RATE = 0.75
    rootdirpath = "./parsed/"
    N_docs = 120
    TITLE_WEIGHT = np.log(N_docs)
    wikipedia_repo = "../gettingArticles/wikipedia_repos/1/"
    USE_TITLES = True

    processes_numb = 4
    dirpath = main(N_docs, rootdirpath, wikipedia_repo,
                   COND_REMOVE_RATE=COND_REMOVE_RATE,
                   ABSOL_REMOVE_RATE=ABSOL_REMOVE_RATE,
                   TITLE_WEIGHT=TITLE_WEIGHT,
                   USE_TITLES=USE_TITLES,
                   processes_numb=processes_numb)
