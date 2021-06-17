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

class NoisyRequester:
    def __init__(self, repo_path="./parsing/parsed/5_120000_wikipedia1/", wiki_repo="./gettingArticles/wikipedia_repos/1/"):
        print("Loading search engine")
        self.repo_path = repo_path
        self.wiki_repo = wiki_repo
        self.titles = None
        self.__load_titles()
        self.W = None
        self.D = None
        self.__load_words()
        self.__load_docs()
        self.A = None
        self.__load_A()

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
        self.A = scipy.sparse.load_npz(self.repo_path + "A.npz")

    def __get_query_vector(self, q):
        q = word_tokenize(q)
        q = [ps.stem(w) for w in q]
        return scipy.sparse.csc_matrix(np.array([1 if (w in q and w not in most_common_words) else 0 for w in self.W])), q


    def make_query(self, q, N_MOST_IMPORTANT=200):

        qv, qs = self.__get_query_vector(q)

        cosfi_ = np.array((qv @ self.A).todense())[0]

        qv_norm = scipy.sparse.linalg.norm(qv)
        if qv_norm == 0:
            return [("", "Nothing found, use other sequence")]
        Div = scipy.sparse.linalg.norm(self.A, axis=0) * qv_norm
        I = np.abs((cosfi_ / Div))

        Importances = [(v, i) for i, v in enumerate(I)]
        Importances.sort(reverse=True)
        Query_res = []
        for e in Importances[:N_MOST_IMPORTANT]:
            with open(self.D[e[1]], encoding="UTF-8") as f:
                Query_res.append((f.readline()[:-1], self.titles[self.D[e[1]]]))

        return Query_res




if __name__ == '__main__':

    R = NoisyRequester()

    while True:
        in_q = input("Pass sht\n")
        if in_q == "__END__":
            break
        print("\n".join([u for u,_ in R.make_query(in_q, N_MOST_IMPORTANT=25)]))

