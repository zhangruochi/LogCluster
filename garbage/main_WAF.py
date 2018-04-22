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



def deal_quotation_mark(s):
    #print(s)
    pattern_one = re.compile("\".*?\"")
    def remove_blank(s):
        s = s.replace(" ","_")
        return s

    middle_content =  list(map(remove_blank,re.findall(pattern_one,s)))
    else_content = re.split(pattern_one,s)

    result = []

    for i in range(len(middle_content)):
        result.append(else_content[i])
        result.append(middle_content[i])
    
    #result.append(else_content[-1]) #换行符 
    return re.split("\s+","".join(result))

# 清洗无用字符,用"_"代替非单词字符
def clear_log(line):

    pattern_one = re.compile("\".+\"")
    pattern_two = re.compile("[0-9]+")
    pattern_three = re.compile("\W+")  # 将词组转换成单词字符

    cleared_day_log = []
    tmp_list = deal_quotation_mark(line)
    #print(tmp_list)
    cleared_day_log = []
    for word in tmp_list:
        word = re.sub(pattern_three, "_", word).strip(" ").strip("_")
        word = re.sub(pattern_two, "0", word)

        if word.startswith("http_agent"):
            word = "http_agent"
        elif word.startswith("http_session_id"):
            word = "http_session_id"    
        elif word.startswith("http_url"):
            word = "http_url"
        elif word.startswith("msg_cookie_name"):
            word = "msg_cookie_name"    


        cleared_day_log.append(word)


    """
    Jul  1 00:14:47 172.21.208.18 date=2017-07-01 time=00:06:17 log_id=20000010 msg_id=000006529800 device_id=FV-1KD3A15800187 vd="root" timezone="(GMT+8:00)Beijing,ChongQing,HongKong,Urumgi" type=attack subtype="waf_signature_detection" pri=alert trigger_policy="" severity_level=Low proto=tcp service=http action=Alert_Deny policy="Zfzx-8080" src=218.95.98.239 src_port=62834 dst=192.168.188.200 dst_port=8080 http_method=get http_url="/manager/html" http_host="219.143.219.157:8080" http_agent="Mozilla/3.0 (compatible; Indy Library)" http_session_id=none msg="[Signatures name: Medium Level Security] [main class name: Bad Robot]: 110000003" signature_subclass="Bad Robot" signature_id="110000003" srccountry="China" content_switch_name="none" server_pool_name="zfzx.clo.com.cn:8080" 

    ['Jul', '1', '00:14:47', '172.21.208.18', 'date=2017-07-01', 'time=00:06:17', 'log_id=20000010', 'msg_id=000006529800', 'device_id=FV-1KD3A15800187', 'vd="root"', 'timezone="(GMT+8:00)Beijing,ChongQing,HongKong,Urumgi"', 'type=attack', 'subtype="waf_signature_detection"', 'pri=alert', 'trigger_policy=""', 'severity_level=Low', 'proto=tcp', 'service=http', 'action=Alert_Deny', 'policy="Zfzx-8080"', 'src=218.95.98.239', 'src_port=62834', 'dst=192.168.188.200', 'dst_port=8080', 'http_method=get', 'http_url="/manager/html"', 'http_host="219.143.219.157:8080"', 'http_agent="Mozilla/3.0_(compatible;_Indy_Library)"', 'http_session_id=none', 'msg="[Signatures_name:_Medium_Level_Security]_[main_class_name:_Bad_Robot]:_110000003"', 'signature_subclass="Bad_Robot"', 'signature_id="110000003"', 'srccountry="China"', 'content_switch_name="none"', 'server_pool_name="zfzx.clo.com.cn:8080"']

    ['Jul', '0', '0_0_0', '0_0_0_0', 'date_0_0_0', 'time_0_0_0', 'log_id_0', 'msg_id_0', 'device_id_FV_0KD0A0', 'vd_root', 'timezone_GMT_0_0_Beijing_ChongQing_HongKong_Urumgi', 'type_attack', 'subtype_waf_signature_detection', 'pri_alert', 'trigger_policy', 'severity_level_Low', 'proto_tcp', 'service_http', 'action_Alert_Deny', 'policy_Zfzx_0', 'src_0_0_0_0', 'src_port_0', 'dst_0_0_0_0', 'dst_port_0', 'http_method_get', 'http_url_manager_html', 'http_host_0_0_0_0_0', 'http_agent_Mozilla_0_0__compatible__Indy_Library', 'http_session_id_none', 'msg_Signatures_name__Medium_Level_Security___main_class_name__Bad_Robot__0', 'signature_subclass_Bad_Robot', 'signature_id_0', 'srccountry_China', 'content_switch_name_none', 'server_pool_name_zfzx_clo_com_cn_0']
    """
    return cleared_day_log


def load_log(filename):

    documents = []
    count = 0
    with open(filename, mode="r", errors="replace") as f:
        while True:
            count += 1
            line = f.readline()
            if len(line) == 0:
                break
            #print(line)
            cleared_line = clear_log(line)
            # print(cleared_line)
            # exit()

            documents.append(" ".join(cleared_line))
            if count % 5000 == 0:
                print(count)

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
    kmeans = KMeans(init="k-means++",n_clusters=n_clusters,n_jobs = -1,random_state=0).fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # node_list = center_of_gravity(kmeans)
    # create_graph(node_list)

    assert(len(labels) == data.shape[0])
    save_txt("centers", centers)

    # save_txt("labels",labels)

    return labels


# 建立数据库
def create_log_tag(labels, filename):
    label_dict = Counter(labels)
    labels_info = sorted(label_dict.items(), key=lambda item: item[1])

    # save_txt("{}_label".format(filename),labels_info)
    save_object("labels_info", labels_info)

    database = defaultdict(list)
    index = 0

    with open(filename, mode="r", errors="replace") as f:
        while True:
            line = f.readline()
            if not line:
                break
            database[labels[index]].append(line)
            index += 1

    save_object("database", database)

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
    tfidf_vectors = tfidf_main(documents)
    labels = kmeans_cluster(tfidf_vectors, clusters)
    database, labels_info = create_log_tag(labels, filename)
    create_view(labels_info, database)


if __name__ == '__main__':
    #filename = "GX03-WAF1000D-1"
    #clusters = 10
    filename = sys.argv[1]
    clusters = int(sys.argv[2])
    main()
    #load_log(filename)

