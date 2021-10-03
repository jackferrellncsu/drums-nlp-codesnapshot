import pandas as pd
import numpy
import pickle
import os
import json
import numpy
import demonstration_routine as helper
import random
from torch.nn import Softmax 
from transformers import BertForMaskedLM, BertTokenizer
from sklearn.model_selection import train_test_split

model =  BertForMaskedLM.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
softmax = Softmax(dim=0)

#-----------------------------Calibration Non-Conformity Score Management---------------------------------#
def mask_data(token_tensor, mask_inds):
    """Replace word at masked id with 103, index for [MASK] token.  Helper function that takes in PyTorch tensor.
    User should avoid calling unless creating own validation set."""
    for i in range(token_tensor.input_ids.shape[0]):
        token_tensor.input_ids[i, mask_inds[i]] = 103

def save_alphas(alphas, filename):
    path = "Data/" + filename + ".txt"
    with open(path, "w") as file:
        json.dump(alphas, file)

def generate_full_alphas(seed = 0):
    """Generates full calibration set from Brown data, in general should not be called by user.
        Seed is set for reproducibility purposes."""

    random.seed(seed)

    brown_data = pd.read_csv("Data/brown_master.csv", sep = "\t")

    train_cal, test = train_test_split(brown_data, test_size=0.25, shuffle=True)
    train, cal = train_test_split(train_cal, test_size=0.08, shuffle=True)

    train_cal = None
    train = None

    cal_clean_sents = cal["clean_sentences"].tolist()
    cal_mask_inds = cal["mask_inds"].tolist()

    alphas_soft = []

    print("Generating Alphas:")

    for j in range(len(cal_clean_sents)):
        input = tokenizer(cal_clean_sents[j], return_tensors = "pt", max_length=256, truncation=True, padding="max_length")
        input["labels"] = input.input_ids.detach().clone()
        mask_data(input, [cal_mask_inds[j]])

        outputs = model(**input)

        result_softmax = softmax(outputs.logits[0][cal_mask_inds[j]]).tolist()

        #Gets predicted prob of true word
        soft_true = result_softmax[input.labels[0][cal_mask_inds[j]]]

        #Gets index of true word
        true_index = input.labels[0][cal_mask_inds[j]]

        alphas_soft.append(1-soft_true)
        if j % 1 == 0:
            print(j)
    save_alphas(alphas_soft, "soft_alphas_full")

def load_cal_alphas(filename):
    """Loads calibration set of non-conformity scores.  User can create their own in .json format, or use 
    those generated from Brown corpus."""
    path = "Data/" + filename + '.txt'
    with open(path, "r") as file:
        loaded_vals = json.load(file)
    return loaded_vals

alphas = load_cal_alphas("soft_alphas_full")

def find_mask_ind(tokenized_sentence):
    """Takes in a PyTorch tensor, returns the index of the masked word in that tensor
    (BERT token 103)."""
    return tokenized_sentence.input_ids[0].tolist().index(103)

def conf_pred(sentence, model = model, conf = .95, calib = alphas, m_ind = -1):
    """sentence is a string containing at least one [MASK] token
        model = bert-base-uncased model from transformers package
        conf = desired level of confidence for resulting interval
        calib = list of non-conformity scores for validation set
        m_ind = index of masked word you would like to predict, default behavior is find the first one
                only change value if dealing with multiple masked words
        
        Takes these things as input, returns an inductive conformal interval with the desired level of confidence.
        Higher level of confidence will generally lead to bigger intervals."""
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