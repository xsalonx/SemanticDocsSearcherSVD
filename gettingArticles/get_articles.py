import random
import os

import requests
from bs4 import BeautifulSoup
import re


class WikipediaArticlesSourcer:
    def __init__(self, wikipedia_repo_path, matchers=None):
        k = int(max([-1] + [int(i) for i in os.listdir(wikipedia_repo_path)])) + 1
        wikipedia_repo_path += f"/{k}"
        os.mkdir(wikipedia_repo_path)
        wikipedia_repo_path += "/"


        self.Q = set()
        self.idxPath = wikipedia_repo_path + "idx.txt"
        self.rawPath = wikipedia_repo_path + "raw_html"
        os.mkdir(self.rawPath)
        self.rawPath += "/"
        self.parsedPath = wikipedia_repo_path + "readable_texts"
        os.mkdir(self.parsedPath)
        self.parsedPath += "/"

        self.i = 0
        self.indexes = {}

        if matchers is not None:
            self.matchers = matchers
        else:
            self.matchers = []
            Matchers_True_req = [lambda s: True]
            Matchers_True_req.append(re.compile("^http[s ]://en.wikipedia.org/wiki/.*[^)]$").match)

            Matchers_False_req = [lambda s: False]
            Matchers_False_req.append(re.compile(".......*[:#].*").match)
            Matchers_False_req.append(re.compile(".*Main_Page").match)

            self.matchers.append(Matchers_True_req)
            self.matchers.append(Matchers_False_req)

    def get_links(self, html, N=50):

        Links = []
        parser = BeautifulSoup(html, "html.parser")

        for a in parser.find_all('a'):
            l = a.get('href')
            if l is not None:
                if l[:4] != "http":
                    l = "https://en.wikipedia.org" + l
                Links.append(l)

        matcher = lambda s: all([m(s) for m in self.matchers[0]]) and all([not nm(s) for nm in self.matchers[1]])
        Links = list(filter(matcher, Links))

        try:
            return random.sample(Links, min(N, len(Links)))
        except:
            return []

    def save_article(self, html, url):

        with open(self.idxPath, "a") as If:
            with open(self.rawPath + f"{self.i}.txt", "w+", encoding="UTF-8") as f:
                print(f"{self.i} :: {url} :: raw html")
                f.write(html)
            with open(self.parsedPath + f"{self.i}.txt", "w+", encoding="UTF-8") as f:
                print(f"{self.i} :: {url} :: parsed")
                parser = BeautifulSoup(html, "html.parser")
                f.write(f"{url}\n\n" + parser.text)
            If.write(f"{url}|{self.i}\n")
            self.indexes[url] = self.i

    def next_article(self):

        if self.Q:
            url = self.Q.pop()
            if url in self.indexes.keys():
                return

            try:
                html = requests.get(url).text
                Links = self.get_links(html)

                if len(self.Q) < 100000:
                    for l in Links:
                        self.Q.add(l)
                self.save_article(html, url)
                self.i += 1
            except:
                print("sth wrong...")

    def get_articles(self, root_url, n):
        self.i = 0
        self.Q = {root_url}
        self.indexes = {}
        while self.Q and self.i < n:
            self.next_article()
            print(f"{len(self.Q)}")



if __name__ == '__main__':
    url = "https://en.wikipedia.org/wiki/Sanskrit"

    wikipedia_repo_path = "./wikipedia_repos"
    wp = WikipediaArticlesSourcer(wikipedia_repo_path=wikipedia_repo_path)

    wp.get_articles(url, n=10)
