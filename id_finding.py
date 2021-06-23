'''
Author: your name
Date: 2021-06-05 19:26:18
LastEditTime: 2021-06-05 21:32:17
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /pg-task11/project/id_finding.py
'''
import numpy as np
import re 
from itertools import combinations
import csv
import os
import json



def write_multi(data,label,mode):
    onehots = []
    for item in label:
        onehot = [0 for i in range(10)]
        labels = [int(t) for t in item.split(',')]
        for key in labels:
            onehot[key-1] = 1
        onehots.append(onehot)
        
    lines = [[data[i]]+onehots[i] for i in range(len(data))]
    with open(f'quda_expert/{mode}.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(['comment_text','Retrieve Value','Filter','Compute Derived Value','Find Extremum','Sort','Determine Range','characterize Distribution','Find Anomalies','Cluster','Correlate'])
        writer.writerows(lines)



if __name__ =='__main__':
    
    with open('./quda_corpus/quda_corpus.txt','r') as f:
        raw = f.readlines()
        
    expert = {(i+1):[] for i in range(20)}

    for i in range(len(raw)):
        start = list(re.finditer(':',raw[i]))[3].span()[0]
        end = raw[i].find(' ')
        expert_id = int(raw[i][start+1:end])
        seq = raw[i]
        expert[expert_id].append(seq)

    if not os.path.exists('quda_expert'):os.mkdir('quda_expert')

    val_texts = []
    val_labels = []
    for key in [2,7,15,17]:
        for line in expert[key]:
            print(line)
            label = line[line.find('[')+1:line.find(']')]
            text = line[line.find(' ')+1:].strip('\n')
            val_texts.append(text)
            val_labels.append(label)

    train_texts = []
    train_labels = []
    for key in set(expert.keys()).difference(set([2,7,15,17])):
        for line in expert[key]:
            label = line[line.find('[')+1:line.find(']')]
            text = line[line.find(' ')+1:].strip('\n')
            train_texts.append(text)
            train_labels.append(label)
    print(val_labels)
    write_multi(val_texts,val_labels,'val')
    write_multi(train_texts,train_labels,'train')



