import numpy as np
import lda
import lda.datasets

titles = lda.datasets.load_reuters_titles()  # 原始的文本资料，长度是395
vocab = lda.datasets.load_reuters_vocab()  # 字典，tuple类型，长度是4258.

x = lda.datasets.load_reuters()  # 加载的one-hot形式的文本资料，是一个（395,4258）的矩阵
print(x)
exit()
model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)  # 设置topic的数量是20，定义模型
model.fit(x)  # 训练模型

topic_word = model.topic_word_  # topic到word的模型，（20,4258）的权重矩阵

n_top_words = 8

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(
        topic_dist)][:-(n_top_words + 1):-1]  # 找到topic对应的前8个最重要的单词
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

doc_topic = model.doc_topic_  # doc到topic的权重矩阵（395,20）
for i in range(10):
    print("{} (top topic: {})".format(
        titles[i], doc_topic[i].argmax()))  # 输出每个文本对应的topic
