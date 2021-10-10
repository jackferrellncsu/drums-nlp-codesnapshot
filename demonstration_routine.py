import json
import numpy
from torch.nn import Softmax

from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
softmax = Softmax(dim=0)

#loads a list from .txt file in json format
def load_cal_alphas(filename):
    path = "Data/" + filename + '.txt'
    with open(path, "r") as file:
        loaded_vals = json.load(file)
    return loaded_vals


#Gets index of masked word from tokenized tensor.
def find_mask_ind(tokenized_sentence):
    return tokenized_sentence.input_ids[0].tolist().index(103)


#Input clean sentences.  
#Words should already be masked.
def conf_pred(sentence, model, conf, calib, m_ind = -1):
    q_soft = numpy.quantile(calib, conf)

    conf_intervals = []

    
    input = tokenizer(sentence, return_tensors='pt')

    #Get index of masked word
    if m_ind == -1:
        m_ind = find_mask_ind(input)
    
    outputs = model(**input)

    result_softmax = softmax(outputs.logits[0][m_ind]).tolist()
    result_softmax = numpy.array(result_softmax)

    non_confs = 1 - result_softmax

    region_inds = numpy.nonzero(non_confs <= q_soft)[0].tolist()

    words = tokenizer.convert_ids_to_tokens(region_inds)

    return words