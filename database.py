#!/usr/bin/env python3

# info
# -name   : zhangruochi
# -email  : zrc720@gmail.com

import pickle
import sys


with open("{}".format(sys.argv[1]), "rb") as f:
    database = pickle.load(f)

with open("{}/labels_info.pkl".format(sys.argv[1].split("/")[0]), "rb") as f:
    label_info = pickle.load(f)

while True:

    print("\n----------------------------------------\n")
    print("\nlog information: \n")
    for item in label_info:
        print("category {}, number: {}".format(item[0], item[1]))

    print("\n----------------------------------------\n")
    print("please input a \"category\" and the \"number\" of logs you want to display: ")
    content = input()

    print("\n----------------------------------------\n")
    print("\nthe example logs is: \n")

    label, num = list(map(int, content.split(" ")))


    if num <= len(database[label]):
        for log in database[label][0:num]:
            print(log)
            
    else:
        for log in database[label]:
            print(log)
    print("\n----------------------------------------\n")  

    print("continue or exit? ")
    flag = input()    
    if flag == "exit":
        break  
        
        
