from jieba import analyse
# 引入TextRank关键词抽取接口


def load_one_doc(path):
    with open(path, mode="r", encoding="utf8") as f:
        doc = f.read()
    return doc


def main():
    analyse.set_stop_words("stopwords.txt")

    path = "docs.txt"
    doc = load_one_doc(path)

    tfidf = analyse.extract_tags

    keywords = analyse.textrank(doc, withWeight=True)
    print("")
    print("TextRank result: ")
    print(keywords)


if __name__ == '__main__':
    main()
