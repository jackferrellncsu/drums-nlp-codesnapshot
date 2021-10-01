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




#initialize pre-trained model
model =  BertForMaskedLM.from_pretrained("bert-base-uncased")
text = "Home entertainment [MASK] up"
confidence = 0.75
m_ind = 3



helper.conf_pred(text, model, confidence, alphas)



