from textrank.TextRankforKeyword import TextRankforKeyword
from textrank.TextRankforSentence import TextRankforSentence


def load_docs(path):
    with open(path, mode="r", encoding="utf8") as f:
        doc = f.read()
    return doc


def main():

    path = "04.txt"
    doc = load_docs(path)

    tr4w = TextRankforKeyword(stop_words_file='./stopwords.txt')

    # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象
    tr4w.analyze(text=doc, window=5)

    print("")
    print('关键词：')
    for item in tr4w.get_keywords(10, word_min_len=1):
        print(item.word, item.weight)

    print()
    print('关键短语：')
    for phrase in tr4w.get_keyphrases(keywords_num=10, min_occur_num=1):
        print(phrase)

    tr4s = TextRankforSentence(stop_words_file='./stopwords.txt')
    tr4s.analyze(text=doc, lower=True, source='all_filters')

    print()
    print('摘要：')
    for item in tr4s.get_key_sentences(num=5):
        print(item.index,item.sentence,item.weight)

if __name__ == '__main__':
    main()