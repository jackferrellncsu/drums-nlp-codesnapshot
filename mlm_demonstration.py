import pandas as pd
import numpy
import pickle
import os
import json
import demonstration_routine as helper


alphas = helper.load_cal_alphas("soft_alphas_0_0")

from transformers import BertTokenizer, BertForMaskedLM
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#initialize pre-trained model
model =  BertForMaskedLM.from_pretrained("bert-base-uncased")

test = tokenizer("Home entertainment [MASK] up", return_tensors='pt')

outputs = model(**test)



#Input clean sentences.  
#Words should already be masked.
def conf_pred(sentence, model, conf, calib):
    q_soft = numpy.quantile(calib, conf)

    conf_intervals = []

    input = tokenizer(sentence, return_tensors= 'pt', max_length=128)
    input['labels'] = input.input_ids.detatch().clone()

    outputs = model(**input)



