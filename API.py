'''
Author: your name
Date: 2021-06-03 22:09:46
LastEditTime: 2021-06-04 21:39:32
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /pg-task11/project/API.py
'''
#! user/bin/python
#-*- coding: UTF-8 -*-

import requests
import pickle
import json
import time
from metrics import accuracy_thresh
import logging
import torch
import os
import csv

api = "https://freenli.projects.zjvis.org/get_sentences_tasks"

def readdata(filepath):
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        labels = [row[1:] for row in reader][1:]
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        texts = [row[0] for row in reader][1:]
    return texts, labels

def writetext(data):
    with open('textdata','w') as f:
        for item in data:
            f.write(item+'\n')


if __name__ == '__main__':
    texts, labels = readdata('/home/pg-task11/project/data_multi/val_data.csv')
    # start = time.time()
    # # for turn, seq  in enumerate(texts):
    # #     params ={'sentences': seq}
    # #     print(params)
    # params = 
    # x = requests.get(url= api, params = params).json()
    # print(x)
    # print("{:-^50s}".format(f"Total time for one iter is {time.time()-start} seconds. "))
    # writetext(texts)
    with open('textdata','r') as f:
        text = f.read()
    params = json.dumps(str(text))
    x = requests.get(url= api, params = params).json()
    print(x)

