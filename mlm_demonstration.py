import pandas as pd
import numpy
import pickle
import os
import json
import numpy
import demonstration_routine as helper
from torch.nn import Softmax 
from transformers import BertForMaskedLM



alphas = helper.load_cal_alphas("soft_alphas_0_0")

test_sent_1 = "...to go with him to a community [MASK]. But when we met, Barack was a community organizer." 
test_sent_2 = "And he urged the people in that meeting in that community to devote themselves to closing the gap between those two ideas, to work together to try to make the world as it is and the world as it should [MASK]one and the same."
test_sent_3 = "And they opened many new doors for millions of female doctors and nurses and artists and authors all of whom have [MASK] [MASK]. And by getting a good education you too can control your own destiny."

test_sent_4 = "And they opened many new doors for millions of female doctors and nurses and artists and authors all of whom have followed [MASK]. And by getting a good education you too can control your own destiny."
test_sent_5 = "And they opened many new doors for millions of female doctors and nurses and artists and authors all of whom have [MASK] them. And by getting a good education you too can control your own destiny."

#initialize pre-trained model
model =  BertForMaskedLM.from_pretrained("bert-base-uncased")
text = "Home entertainment [MASK] up"
confidence = 0.75
m_ind = 3

helper.conf_pred(test_sent_1, model, confidence, alphas)

helper.conf_pred(test_sent_2, model, confidence, alphas)

helper.conf_pred(test_sent_3, model, confidence, alphas)
helper.conf_pred(test_sent_3, model, confidence, alphas, m_ind = 23)

helper.conf_pred(test_sent_4, model, confidence, alphas)
helper.conf_pred(test_sent_5, model, confidence, alphas)

