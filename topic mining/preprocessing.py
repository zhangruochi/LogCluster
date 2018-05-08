import jieba
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

import sys
from gensim import corpora, models


def load_docs(path):
    docs = []
    with open(path, mode="r", encoding="utf8") as f:
        while(True):
            doc = f.readline()
            if doc:
                doc = doc.strip("\n").strip("。")
                if len(doc) > 0:
                    docs.append(doc)
            else:
                break
    return docs[0:-1]


def segmentation(docs, stop_words_set):
    corpus = []
    for doc in docs:
        seg_list = jieba.cut(doc, cut_all=False)
        corpus.append(
            [term for term in seg_list if str(term) not in stop_words_set])
    return corpus


def get_stop_words_set(file_name):
    with open(file_name, 'r', encoding="utf8") as file:
        return set([line.strip() for line in file])


def lda(corpus):
    word_dict = corpora.Dictionary(corpus)  # 生成文档的词典，每个词与一个整型索引值对应
    corpus_list = [word_dict.doc2bow(text)
                   for text in corpus]  # 词频统计，转化成空间向量格式
    lda = models.ldamodel.LdaModel(
        corpus=corpus_list, id2word=word_dict, num_topics=5, alpha='auto')

    for pattern in lda.show_topics():
        print(pattern)
        print("")

    """
    output_file = './lda_output.txt'
    with open(output_file, mode = 'w', encoding = "utf8") as f:
        for pattern in lda.show_topics():
            f.write("%s" % str(pattern))
    """


def main():
    docs_path = "docs.txt"
    docs = load_docs(docs_path)
    stop_words_path = "stopwords.txt"
    stop_words = get_stop_words_set(stop_words_path)

    corpus = segmentation(docs, stop_words)
    # print(corpus)
    lda(corpus)


if __name__ == '__main__':
    main()
