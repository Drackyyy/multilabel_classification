'''
Author: your name
Date: 2021-05-28 20:27:36
LastEditTime: 2021-06-23 14:26:09
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /project/fast-bert/fast_bert/main.py
'''
from data_cls import BertDataBunch
from learner_cls import BertLearner
from metrics import accuracy_thresh
import logging
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
databunch = BertDataBunch('./quda_expert/', './label/',
                          tokenizer='./pretrained_model',
                          train_file='train.csv',
                          val_file='val.csv',
                          label_file='labels.csv',
                          text_col='comment_text',
                          label_col=['Retrieve Value','Filter','Compute Derived Value','Find Extremum','Sort','Determine Range','characterize Distribution','Find Anomalies','Cluster','Correlate'],
                          batch_size_per_gpu=64,
                          max_seq_length=41,
                          multi_gpu=True,
                          multi_label=True,
                          model_type='bert')

logger = logging.getLogger()
device_cuda = torch.device("cuda")
metrics = [{'name': 'accuracy_thresh', 'function': accuracy_thresh}]

learner = BertLearner.from_pretrained_model(
						databunch,
						pretrained_path='./pretrained_model',
						metrics=metrics,
						device=device_cuda,
						logger=logger,
						output_dir='learner_cls_output_expert',
						finetuned_wgts_path=None,
						warmup_steps=0,
						multi_gpu=True,
						is_fp16=True,
						multi_label=True,
						logging_steps=10)


learner.fit(epochs=5,
			lr=5e-4,
			validate=True, 	# Evaluate the model after each epoch
			schedule_type=None,
			optimizer_type="lamb")

learner.save_model()
