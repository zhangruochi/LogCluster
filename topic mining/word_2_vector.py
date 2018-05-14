from gensim.models import word2vec
import logging

# 训练主程序
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
# 加载语料
sentences = word2vec.Text8Corpus(u'model/text8')
# 训练模型  skip-gram
model = word2vec.Word2Vec(sentences, size=200, window=10,
                          min_count=64, sg=1, hs=1, iter=10, workers=25)

