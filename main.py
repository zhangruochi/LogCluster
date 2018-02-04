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


#import networkx as nx
#import matplotlib.pyplot as plt



#保存和读取中间文件
def save_object(name,object):
    with open("{}_{}_cache/{}.pkl".format(filename,clusters,name),"wb") as f:
        pickle.dump(object,f)

def read_object(filename):
    with open(filename,"rb") as f:
        object = pickle.load(f)
    return object


def save_txt(name,object):
    with open("{}_{}_cache/{}.txt".format(filename,clusters,name),"w") as f:
        for item in object:
            f.write(str(item))
            f.write("\n")    




# 清洗无用字符,用"_"代替非单词字符
def clear_log(day_log):
    pattern_one = re.compile("\W")  # 将词组转换成单词字符
    pattern_two = re.compile("[0-9]+")

    cleared_day_log = []
    for chars in day_log:
        tmp = re.sub(pattern_one, "_", chars).strip("_")
        if tmp.startswith("the_system_clock"):
            tmp = "the_system_clock"

        cleared_day_log.append(re.sub(pattern_two, "0", tmp))

    # 去掉时间特征
    #cleared_day_log = cleared_day_log[2:9] + cleared_day_log[10:]
    """
    ['Jul  1 00:00:05 2017-06-30 16: 00:05 GX01-USG5530-1 %%01SEC/4/POLICYDENY(l): protocol=1', ' source-ip=10.38.31.9', ' source-port=0', ' destination-ip=10.192.0.9', ' destination-port=0', ' time=2017/07/01 00:00:05', ' interzone-untrust(public)-trust(public) inbound', ' policy=3.\n']
    ['Jul_1_00_00_05', '2017_06_30_16__00_05', '01SEC_4_POLICYDENY_l', 'protocol_1', 'source_ip_10_38_31_9', 'source_port_0', 'destination_ip_10_192_0_9', 'destination_port_0', 'time_2017_07_01_00_00_05', 'interzone_untrust_public__trust_public__inbound', 'policy_3']
    
    ['GX01_USG5530_1', '01SEC_4_POLICYDENY_l', 'protocol_1', 'source_ip_10_38_31_9', 'source_port_0', 'destination_ip_10_192_0_9', 'destination_port_0', 'interzone_untrust_public__trust_public__inbound', 'policy_3']
    """

    # print(cleared_day_log)
    #['Jul_0_00_00_00', '0000_00_00_00__00_00', 'GX00_USG0000_0', '00SEC_0_POLICYDENY_l', 'protocol_0', 'source_ip_00_00_00_0', 'source_port_0', 'destination_ip_00_000_0_0', 'destination_port_0', 'time_0000_00_00_00_00_00', 'interzone_untrust_public__trust_public__inbound', 'policy_0']
    # exit()

    return cleared_day_log


def load_log(filename):

    documents = []
    #count = 0
    with open(filename, mode="r", errors="replace") as f:
        while True:
            #count += 1
            line = f.readline().split(",")
            head = line[0].split()

            if line == [""]:
                break
            # print(line)

            row = []
            row.append("_".join(head[0:3]))
            row.append("_".join(head[3:6]))
            try:
                row.append(head[6])
            except:
                print(line)
                print(count)
                exit()

            middle = " ".join(head[7:]).split(":")

            row.append("_".join(middle[:-1]))
            row.append(middle[-1])
            row += line[1:]

            # print(row)
            cleared_row = clear_log(row)
            # print(cleared_row)
            # exit()

            documents.append(" ".join(cleared_row))

    # print(len(documents))
    # 200688

    return documents


# 使用 tf——idf 将文本处理成词频向量
def tfidf_main(documents):

    vectorizer = CountVectorizer()
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

    save_txt("words",words)
    
    return tfidf_vectors


# 计算两个向量之间的距离
def distance(a, b):
    sum = 0
    for i, j in zip(a, b):
        sum += (abs(i - j) ** 2)

    return sum ** 0.5


#创建以重心为点，重心间的距离为边的图
def create_graph(node_list):
    G = nx.Graph() 
    G.add_weighted_edges_from(node_list)
    nx.draw(G,pos = nx.random_layout(G),node_color = 'b',edge_color = 'r',with_labels = True,font_size =18,node_size =20)
    plt.savefig("graph.png")
    #plt.show()


# processing center of gravity
def center_of_gravity(kmeans):
    centers = kmeans.cluster_centers_
    node_list = []
    for i in range(len(centers) - 1):
        for j in range(i + 1, len(centers)):
            node_list.append((i, j, distance(centers[i], centers[j])))

    return node_list 


# 使用聚类算法
def kmeans_cluster(data, n_clusters=10):
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, random_state=0).fit(data)
    labels = kmeans.labels_
    node_list = center_of_gravity(kmeans)
    
    #create_graph(node_list)

    assert(len(labels) == data.shape[0])

    #save_txt("labels",labels)

    return labels


#建立数据库
def create_log_tag(labels,filename):
    label_dict = Counter(labels)
    sorted_labels = sorted(label_dict.items(), key=lambda item: item[1])
    
    #save_txt("{}_label".format(filename),sorted_labels)
    save_object("label",sorted_labels)

    log_dict = defaultdict(list) 
    index = 0

    with open(filename, mode="r", errors="replace") as f:
        while True:
            line = f.readline()
            if not line:
                break
            log_dict[labels[index]].append(line)    
            index+= 1 

    save_object("database",log_dict) 

    return log_dict

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

def main():

    #data = load_log(filename)
    # kmeans_cluster(data)
    if not os.path.exists("{}_{}_cache".format(filename,clusters)):
        os.mkdir("{}_{}_cache".format(filename,clusters))

    documents = load_log(filename)
    tfidf_vectors = tfidf_main(documents)
    labels = kmeans_cluster(tfidf_vectors,n_clusters=clusters)
    log_dict = create_log_tag(labels,filename)

    print('database creating successful......')


if __name__ == '__main__':
    filename = "GX01-USG5530-1"
    clusters = 10
    #filename = sys.argv[1]
    #clusters = int(sys.argv[2])
    main()
