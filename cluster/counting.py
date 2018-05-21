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

from operator import itemgetter

from collections import Counter
from collections import defaultdict
from collections import OrderedDict

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
        for key, value in cf.items("COUNTING"):
            option_dict[key] = eval(value)

        return option_dict


def get_chars(raw_chars):
    if not raw_chars:
        return ["-", "\\.", "\\/", "%"]

    else:
        ZHUAN = "*.?+$^[](){}|\/"
        result = []
        for char in raw_chars:
            if char in ZHUAN:
                result.append("\{}".format(char))
            else:
                result.append(char)
    return result


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
            # print(cleared_line)
            # exit()

            documents.append(cleared_line.split(" "))
            if count % 5000 == 0:
                print(count)

    # print(len(documents))
    # 200688

    return documents


def all_words_counting(documents):
    content = []
    for log in documents:
        content += log

    count_dict = Counter(content)

    ordered_count_dict = OrderedDict(
        sorted(count_dict.items(), key=itemgetter(1)))
    # save_object("words_counting",ordered_count_dict)

    return ordered_count_dict


def is_abnormal(vector, threshold, num):

    tmp_count = 0
    for number in vector:
        if number <= threshold:
            tmp_count+=1
    if tmp_count > num:
        return True        
            
    return False


def get_abnormal(counts_vectors, threshold, num):
    results = []
    index_list = []
    for index,vector in enumerate(counts_vectors):
        if is_abnormal(vector, threshold,num):
            results.append(vector)
            index_list.append(index)
    return results,index_list


def convert_to_counts_vectors(documents, rf_dict):
    counts_vectors = []
    for log in documents:
        count_vector = []
        for word in log:
            count_vector.append(rf_dict[word])
        counts_vectors.append(count_vector)
    return counts_vectors



# 使用 tf——idf 将文本处理成词频向量
def tf_main(documents):
    threshold = 10 #define
    num = 5 #times

    ordered_count_dict = all_words_counting(documents)
    #mydict_new = dict([val,key] for key,val in ordered_count_dict.items())

    counts_vectors = convert_to_counts_vectors(documents, ordered_count_dict)
    abnormals,index_list = get_abnormal(counts_vectors,threshold, num)

    return abnormals,index_list


def main():

    if not os.path.exists("{}_{}_cache".format(filename, clusters)):
        os.mkdir("{}_{}_cache".format(filename, clusters))

    documents = load_log(filename)
    # all_words_counting(documents)
    print("preprocessing finished......\n")

    abnormals,index_list = tf_main(documents)
    for index,log_vector in zip(index_list,abnormals):
        print(log_vector)
        print(" ".join(documents[index]))
        print("")

if __name__ == '__main__':
    option_dict = get_options()

    filename = option_dict['filename']
    clusters = option_dict['clusters']
    chars = option_dict['chars']
    num = option_dict['num']

    main()
