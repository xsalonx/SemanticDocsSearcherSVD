import os



if __name__ == '__main__':

    parsedpath = "../parsing/parsed"
    path = "./gettingArticles/wikipedia_repos/1/readable_texts/"
    for d in os.listdir(parsedpath):
        IDpath = parsedpath + "/" + d + "/IDX_TO_DOCS.txt"
        with open(IDpath) as f:
            L = len(f.readlines())
        with open(IDpath, "w") as f:
            f.write("\n".join([f"{path}{i}.txt" for i in range(L)]))