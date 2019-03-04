#!/usr/bin/env python3

# info
# -name   : zhangruochi
# -email  : zrc720@gmail.com


import pandas as pd
import numpy as np
import re
import os
import pickle
import sys

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

from collections import Counter
from collections import defaultdict

try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser


#import networkx as nx
#import matplotlib.pyplot as plt


def get_options():
    cf = ConfigParser.ConfigParser()

    if os.path.exists("config.cof"):
        cf.read('config.cof')
    else:
        print("there is no config.cof!")
        exit()

    option_dict = dict()
    for key, value in cf.items("CLUSTER"):
        option_dict[key] = eval(value)

    return option_dict


"""
def convert_chars(raw_chars):
    chars = []
    for char in raw_chars:
        if char == ".":
            chars.append("\.")
        elif char == "%":
            chars.append("%%")
        else:
            chars.append(char)
    return chars        



def get_chars(raw_chars):
    if not raw_chars:
        return ["-","\\.","\\/"]

    else:    
        ZHUAN = "*.?+$^[](){}|\/"
        result = []
        for char in raw_chars:
            if char in ZHUAN:
                result.append("\{}".format(char))
            else:
                result.append(char)
    return result            
"""

# 保存和读取中间文件


def save_object(name, object):
    with open("{}_{}_cache/{}.pkl".format(filename, clusters, name), "wb") as f:
        pickle.dump(object, f)


def read_object(filename):
    with open(filename, "rb") as f:
        object = pickle.load(f)
    return object


def save_txt(name, object):
    with open("{}_{}_cache/{}.txt".format(filename, clusters, name), "w") as f:
        for item in object:
            f.write(str(item))
            f.write("\n")


# 清洗无用字符,用"_"代替非单词字符
def clear_log(line, chars):

    pattern_one = re.compile("[0-9]+")
    pattern_two = re.compile("|".join(chars))  # 可以给定需要保留的参数

    tmp_line = re.sub(pattern_one, "0", line)
    clearned_line = re.sub(pattern_two, "_", tmp_line).strip(" ").strip("_")

    return clearned_line


def load_log(filename):

    documents = []
    count = 0
    with open(filename, mode="r", errors="replace") as f:
        while True:
            count += 1
            line = f.readline()
            if len(line) == 0:
                break
            # print(line)
            cleared_line = clear_log(line, chars)

            documents.append(cleared_line)
            if count % 5000 == 0:
                print(count)

    # print(len(documents))
    # 200688

    return documents


# 使用 tf——idf 将文本处理成词频向量
def tfidf_main(documents, max, min):

    vectorizer = CountVectorizer(max_df=max, min_df=min)
    transformer = TfidfTransformer()
    tf = vectorizer.fit_transform(documents)
    tfidf_vectors = transformer.fit_transform(tf)

    words = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    weight = tfidf_vectors.toarray()

    """
    for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重   
        for j in range(len(words)):  
            print words[j],weight[i][j] 
    """

    save_txt("words", words)
    print("preprocessing finished......")

    return tfidf_vectors


# 计算两个向量之间的距离
def distance(a, b):
    sum = 0
    for i, j in zip(a, b):
        sum += (abs(i - j) ** 2)

    return sum ** 0.5


# 创建以重心为点，重心间的距离为边的图
def create_graph(node_list):
    G = nx.Graph()
    G.add_weighted_edges_from(node_list)
    nx.draw(G, pos=nx.random_layout(G), node_color='b', edge_color='r',
            with_labels=True, font_size=18, node_size=20)
    plt.savefig("graph.png")
    # plt.show()


# processing center of gravity
def center_of_gravity(kmeans):
    centers = kmeans.cluster_centers_
    node_list = []
    for i in range(len(centers) - 1):
        for j in range(i + 1, len(centers)):
            node_list.append((i, j, distance(centers[i], centers[j])))

    return node_list


# 使用聚类算法
def kmeans_cluster(data, n_clusters):
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters,
                    n_jobs=-1, random_state=0).fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # node_list = center_of_gravity(kmeans)
    # create_graph(node_list)

    assert(len(labels) == data.shape[0])
    #save_txt("centers", centers)

    # save_txt("labels",labels)

    return labels


def create_sub_files(database):
    if not os.path.exists("{}_{}_cache/cluster_files".format(filename, clusters)):
        os.mkdir("{}_{}_cache/cluster_files".format(filename, clusters))

    for key, value in database.items():
        with open("{}_{}_cache/cluster_files/cluster_{}.log".format(filename, clusters, key), "w") as f:
            for log in value:
                f.write(log)


# 建立数据库
def create_log_tag(labels, filename):
    label_dict = Counter(labels)

    labels_info = sorted(label_dict.items(), key=lambda item: item[1])
    # save_txt("{}_label".format(filename),labels_info)
    # save_object("labels_info", labels_info)

    database = defaultdict(list)
    index = 0

    with open(filename, mode="r", errors="replace") as f:
        while True:
            line = f.readline()
            if not line:
                break
            database[labels[index]].append(line)
            index += 1

    #save_object("database", database)
    create_sub_files(database)

    return database, labels_info


"""
def get_exception_record(filename, lower_freq_labels, labels):
    masks = []
    for label in labels:
        if label in lower_freq_labels:
            masks.append(1)
        else:
            masks.append(0)

        with open(filename, mode="r", errors="replace") as log_f:
            for decision in masks:
                line = log_f.readline()
                if decision:
                    ex_f.write(line)
"""


def create_view(labels_info, database):
    num = 5
    with open("{}_{}_cache/result.txt".format(filename, clusters), "w") as f:
        f.write("log information: \n\n")
        for item in labels_info:
            f.write("category {}, number: {}\n".format(item[0], item[1]))
        f.write("\n----------------------------------------\n")
        f.write("\nthe example logs are: \n")

        for category in labels_info:
            f.write("\ncategory {}\n".format(category[0]))
            if num <= len(database[category[0]]):
                for log in database[category[0]][0:num]:
                    f.write(log)
            else:
                for log in database[category[0]]:
                    f.write(log)


def main():

    # data = load_log(filename)
    # kmeans_cluster(data)
    if not os.path.exists("{}_{}_cache".format(filename, clusters)):
        os.mkdir("{}_{}_cache".format(filename, clusters))

    documents = load_log(filename)
    tfidf_vectors = tfidf_main(documents, max, min)
    labels = kmeans_cluster(tfidf_vectors, clusters)
    database, labels_info = create_log_tag(labels, filename)
    create_view(labels_info, database)


if __name__ == '__main__':

    option_dict = get_options()
    filename = option_dict['filename']
    clusters = option_dict['clusters']
    max = option_dict['max']
    min = option_dict['min']
    chars = option_dict['chars']

    main()
