import sys


def make_titles_file(path):

    T = []
    with open(path + "/titles.txt", "w", encoding="UTF-8") as f:
        with open(path + "/idx.txt") as g:
            L = len(g.readlines())
        for i in range(L):
            print(f"{i}/{L}")
            p = path + f"/readable_texts/{i}.txt"
            with open(p, encoding="UTF-8") as g:
                g.readline()
                title = g.readline()
                while title == "\n":
                    title = g.readline()
                title = title.split("-")[:-1]
                if len(title) == 1:
                    title = title[0]
                else:
                    title = "-".join(title)
                T.append((p, title))
        f.write("\n".join([f"{p[1:]}|{t}" for (p, t) in T]))

if __name__ == '__main__':
    make_titles_file(sys.argv)