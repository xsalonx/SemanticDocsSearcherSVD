import re
import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import scipy.linalg
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

ps = PorterStemmer()
import shutil
import os
most_common_words = set(stopwords.words('english'))

with open("./parsing/most_common", encoding="UTF-8") as f:
    t = f.read()
    sw = t.split("\n")
    for w in sw:
        most_common_words.add(w)

class Requester:
    def __init__(self, k=615, repo_path="./parsing/parsed/5_120000_wikipedia1/", wiki_repo="./gettingArticles/wikipedia_repos/1/"):
        self.repo_path = repo_path
        self.wiki_repo = wiki_repo
        self.titles = None
        self.__load_titles()
        self.k = k
        self.W = None
        self.D = None
        self.__load_words()
        self.__load_docs()
        self.U = None
        self.s = None
        self.Vt = None
        self.S = None

        print("Loading search engine")
        dirpath = self.repo_path + "svd/" + str(k)
        if str(k) not in os.listdir(self.repo_path + "svd"):
            raise ValueError(f"No svd for k={k}\n")
            # A = self.__load_A()
            # self.U, self.s, self.Vt = scipy.sparse.linalg.svds(A, k=k, which='LM')
            #
            # os.mkdir(dirpath)
            # dirpath += "/"
            # np.save(dirpath + "U", self.U)
            # np.save(dirpath + "s", self.s)
            # np.save(dirpath + "Vt", self.Vt)
        else:
            dirpath += "/"
            self.U = np.load(dirpath + "U.npy")
            self.s = np.load(dirpath + "s.npy")
            self.Vt = np.load(dirpath + "Vt.npy")
        self.S = np.diag(self.s) @ self.Vt
        print("Requester is ready")

    def __load_titles(self):
        with open(self.wiki_repo + "titles.txt", encoding="UTF-8") as f:
            L = f.read().split("\n")
            LL = [e.split("|") for e in L]
            self.titles = dict(LL)

    def __load_words(self):
        with open(self.repo_path + "IDX_TO_WORD.txt", encoding="UTF-8") as f:
            T = f.read()
            self.W = T.split("\n")

    def __load_docs(self):
        with open(self.repo_path + "IDX_TO_DOCS.txt", encoding="UTF-8") as f:
            T = f.read()
            self.D = T.split("\n")

    def __load_A(self):
        return scipy.sparse.load_npz(self.repo_path + "A.npz")

    def __get_query_vector(self, q):
        q = word_tokenize(q)
        q = [ps.stem(w) for w in q]
        return np.array([1 if (w in q and w not in most_common_words) else 0 for w in self.W]), q

    def get_k(self):
        return self.k

    def make_query(self, q, N_MOST_IMPORTANT=500):

        qv, qs = self.__get_query_vector(q)

        cosfi_ = qv @ self.U @ self.S
        qv_norm = np.linalg.norm(qv)
        if qv_norm == 0:
            return [("", "Nothing found, use other sequence")]
        Div = np.linalg.norm(self.S, axis=0) * qv_norm


        Importances = [(v, i) for i, v in enumerate(np.abs(cosfi_ / Div))]
        Importances.sort(reverse=True)
        Query_res = []

        for e in Importances[:N_MOST_IMPORTANT]:
            with open(self.D[e[1]], encoding="UTF-8") as f:

                Query_res.append((f.readline()[:-1], self.titles[self.D[e[1]]]))

        return Query_res




# if __name__ == '__main__':
#
#     R = Requester(k=615)
#
#     while True:
#         in_q = input("Pass sht\n")
#         if in_q == "__END__":
#             break
#         print("\n".join([u for u,_ in R.make_query(in_q, N_MOST_IMPORTANT=25)]))

