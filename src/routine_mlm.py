from os import getpid
import pandas as pd
import random
import json
import numpy 
import sys, getopt
import pickle

#--------------------------Data Prep Helpers--------------------------------------------------#
#Removes unecessary tags, i.e. -tl, fw-, etc.
def fix_pos(tag):
    if not tag: 
        return None
    s = tag.split("-")
    if s[0] == 'fw':
        return s[1]
    else:
        return s[0]

#load in sentences df
#Not put back together yet, in (pre-bert) token form
def load_sentencesdf():
    raw_data = pd.read_csv("Data/brown.csv")
    sentences = raw_data["raw_text"].str.split(expand=True)

    pos = sentences.apply(lambda x: x.str.rsplit("/").str[1])
    pos = pos.applymap(lambda x: fix_pos(x))

    sentences = sentences.apply(lambda x: x.str.rsplit("/").str[0])
    sentences.insert(180, "mask_ind", None)
    return sentences, pos

#Get index we will later mask, only called in 
def get_mask_helper(row):
    #end and beginning offset by one due to [CLS] token when making bert embeddings
    range = row.notna().sum()+1
    ind = random.randrange(1, range)

    return ind

#puts sentences back together, returns a list of them
def whole_sentences(sentences):
    clean_sentences = sentences.apply(lambda x: x.str.cat(sep=" "), axis = 1)
    return clean_sentences

#gets index we will mask for each sentence
#one word is masked in each sentence 
def get_masked_inds(sentences):
    return sentences.apply(lambda x: get_mask_helper(x), axis = 1)
    
#-------------------------Actual Model Helpers------------------------------------------------#
def mask_data(token_tensor, mask_inds):
    #replace word @masked id with 103, index for [MASK] token 
    for i in range(token_tensor.input_ids.shape[0]):
        token_tensor.input_ids[i, mask_inds[i]] = 103


#Saves a list to a file in json format
def save_alphas(alphas, filename):
    path = "Snapshot_Outs/OutFiles_mlm/" + filename
    with open(path, "w") as file:
        json.dump(alphas, file)

#loads a list from .txt file in json format
def load_cal_alphas(filename):
    path = "Snapshot_Outs/OutFiles_mlm/" + filename
    with open(path, "r") as file:
        loaded_vals = json.load(file)
    return loaded_vals


#Takes in numpy array of nonconformity scores (except for true word) and calibration noncomf scores
#Returns sum of their p-values
def sum_all_pvals(nonconfs, cal_alphas):
    sum = 0
    for i in range(len(nonconfs)):
        sum += len(numpy.nonzero(cal_alphas > nonconfs[i])[0].tolist()) / len(cal_alphas)
    
    return sum

def command_line_seed(argv):
    seed = 0
    try:
        opts, args = getopt.getopt(argv, "s:", ["seed="])
    except getopt.GetoptError:
        print("arguments_test.py -s <seed>")
        sys.exit(2)

    print(opts)
    for opt, arg in opts:
        if opt in ("-s", "--seed"):
            seed = arg
    return seed
    
def save_obj(obj, name):
    with open("Snapshot_Outs/OutFiles_mlm/" + name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open("Snapshot_Outs/OutFiles_mlm/" + name + ".pkl", "rb") as f:
        return pickle.load(f)