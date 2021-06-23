'''
Author: your name
Date: 2021-05-30 14:08:44
LastEditTime: 2021-06-06 12:50:07
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /pg-task11/project/prediction.py
'''
import os
from metrics import metrics_random
import torch
from data_cls import BertDataBunch
from learner_cls import BertLearner
from metrics import accuracy_thresh, metrics_expert
import csv
from transformers import AutoTokenizer

import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
metrics = [{'name': 'accuracy_thresh', 'function': accuracy_thresh}]


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
class BertClassificationPredictor(object):
    def __init__(
        self,
        model_path,
        label_path,
        multi_label=True,
        model_type="bert",
        use_fast_tokenizer=False,
        do_lower_case=True,
        device=None,
    ):
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model_path = model_path
        self.label_path = label_path
        self.multi_label = multi_label
        self.model_type = model_type
        self.do_lower_case = do_lower_case
        self.device = device

        # Use auto-tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=use_fast_tokenizer
        )

        self.learner = self.get_learner()

    def get_learner(self):
        databunch = BertDataBunch(
            self.label_path,
            self.label_path,
            self.tokenizer,
            train_file=None,
            val_file=None,
            batch_size_per_gpu=32,
            max_seq_length=41,
            multi_gpu=False,
            multi_label=self.multi_label,
            model_type=self.model_type,
            no_cache=True,
        )

        learner = BertLearner.from_pretrained_model(
            databunch,
            self.model_path,
            metrics= metrics,
            device=self.device,
            logger=None,
            output_dir=None,
            warmup_steps=0,
            multi_gpu=False,
            is_fp16=False,
            multi_label=self.multi_label,
            logging_steps=0,
        )

        return learner

    def predict_batch(self, texts):
        return self.learner.predict_batch(texts)

    def predict(self, text):
        predictions = self.predict_batch([text])[0]
        return predictions

def readdata(filepath):
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        labels = [row[1:] for row in reader][1:]
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        texts = [row[0] for row in reader][1:]
    return texts, labels

def process(path,multiple_predictions):
    outputs = []
    reals = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        order = list(reader)[0][1:]
    for item in multiple_predictions:
        output = [0 for i in range(10)]
        for i in range(len(item)):
            pos = order.index(item[i][0])
            output[pos] = item[i][1]
        output = [tem > THRESH for tem in output]
        outputs.append(output)
    outputs = torch.tensor(outputs)
    for item in labels:
        reals.append(list(map(int,item)))
    reals = torch.tensor(reals)
    return outputs,reals

if __name__ == '__main__':
    
    THRESH = 0.5
    
    predictor = BertClassificationPredictor(
                    model_path= './learner_cls_output_expert/model_out',
                    label_path='./label', # location for labels.csv file
                    multi_label=True,
                    model_type='bert',
                    do_lower_case=True,
                    device=None) # set custom torch.device, defaults to cuda if available

    texts, labels = readdata('quda_expert/val.csv')
    multiple_predictions = predictor.predict_batch(texts)

    outputs_expert,reals_expert = process('quda_expert/val.csv',multiple_predictions)
    print(metrics_random(outputs_expert,reals_expert))
    

        
        
        
        
    