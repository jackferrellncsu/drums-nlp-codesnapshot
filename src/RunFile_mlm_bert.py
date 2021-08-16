import HelpFile_mlm as helper
import sys, getopt

seed = helper.command_line_seed(sys.argv[1:])

from re import A
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.nn import Softmax
import statistics
import json
import numpy
import pickle



brown_data = pd.read_csv("Data/brown_master.csv", sep = "\t")


train_cal, test = train_test_split(brown_data, test_size=0.25, shuffle=True)
train, cal = train_test_split(train_cal, test_size=0.03, shuffle=True)
train_cal = None
train = None

cal_clean_sents = cal["clean_sentences"].tolist()
cal_mask_inds = cal["mask_inds"].tolist()

# cal_cs_red = cal_clean_sents[1:10]
# cal_mi_red = cal_mask_inds[1:10]

from transformers import BertTokenizer, BertForMaskedLM
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#Initialize pretrained model
model =  BertForMaskedLM.from_pretrained("bert-base-uncased")
model.eval()
#Define softmax function
softmax = Softmax(dim=0)


#Below for-loop calculates the nonconformity scores one at a time bc my stack overflows otherwise
#very slow, be warned
#Non conformity score is 1-softmax of result vector at true word, IE how likely BERT thought the true word was to occur
alphas_soft = []

for i in range(len(cal_clean_sents)):
    input = tokenizer(cal_clean_sents[i], return_tensors = "pt", max_length=256, truncation=True, padding="max_length")
    input["labels"] = input.input_ids.detach().clone()
    helper.mask_data(input, [cal_mask_inds[i]])

    outputs = model(**input)

    result_softmax = softmax(outputs.logits[0][cal_mask_inds[i]]).tolist()

    #Gets predicted prob of true word
    soft_true = result_softmax[input.labels[0][cal_mask_inds[i]]]

    #Gets index of true word
    true_index = input.labels[0][cal_mask_inds[i]]

    alphas_soft.append(1-soft_true)
    if i % 100 == 0:
        print(i)

#-----------------------------Working with conformal predictions-------------------------------------------------------#
alphas_soft[:5]

#Save calibrated alphas
helper.save_alphas(alphas_soft, "soft_alphas1k.txt")

from numpy import quantile

test_clean_sents = test["clean_sentences"].tolist()
test_mask_inds = test["mask_inds"].tolist()

alphas_soft = numpy.array(alphas_soft)

q_soft_90 = quantile(alphas_soft, 0.9)
q_soft_95 = quantile(alphas_soft, 0.95)
q_soft_80 = quantile(alphas_soft, .8)
q_soft_75 = quantile(alphas_soft, .75)
conf_intervals_soft_95 = []
conf_intervals_soft_90 = []
conf_intervals_soft_80 = []
conf_intervals_soft_75 = []

accuracy_95 = 0
accuracy_90 = 0
accuracy_80 = 0
accuracy_75 = 0

point_accuracy = 0

true_inds = []

#Diagnostic Measures:
#contains p-values for all true predictions
p_vals_true = 0
p_vals_false = 0
creds = 0

for i in range(len(test_clean_sents[0:1000])):
    input = tokenizer(test_clean_sents[i], return_tensors = "pt", max_length=128, truncation=True, padding="max_length")
    input["labels"] = input.input_ids.detach().clone()
    helper.mask_data(input, [test_mask_inds[i]])

    #Feed input into model
    outputs = model(**input)

    #Get output at the masked index, apply softmax
    result_softmax = softmax(outputs.logits[0][test_mask_inds[i]]).tolist()
    result_softmax = numpy.array(result_softmax)

    #Predicted word- word with highest softmax score
    
    
    #Do 1-all softmaxes to get nonconf scores
    non_confs = 1 - result_softmax

    #Get nonconf score for true and predicted:
    true_ind = input.labels[0][test_mask_inds[i]]
    pred_ind = numpy.argmax(result_softmax)

    true_nonconf = non_confs[true_ind]
    pred_nonconf = non_confs[pred_ind]

    #Find where nonconf is less than the epsilon quantile:
    region_inds_95 = numpy.nonzero(non_confs <= q_soft_95)[0].tolist()
    region_inds_90 = numpy.nonzero(non_confs <= q_soft_90)[0].tolist()
    region_inds_80 = numpy.nonzero(non_confs <= q_soft_80)[0].tolist()
    region_inds_75 = numpy.nonzero(non_confs <= q_soft_75)[0].tolist()

    conf_intervals_soft_95.append(region_inds_95)
    conf_intervals_soft_90.append(region_inds_90)
    conf_intervals_soft_80.append(region_inds_80)
    conf_intervals_soft_75.append(region_inds_75)

    #Get true label index:
    
    true_soft = result_softmax[true_ind]

    #Add p-value of prediction for true val
    p_vals_true += len(numpy.nonzero(alphas_soft > true_nonconf)[0].tolist()) / len(alphas_soft)

    #Add p-values of false predictions
    p_vals_false = helper.sum_all_pvals(numpy.delete(non_confs, true_ind), alphas_soft)

    #Add p-value of highest softmax:
    creds += len(numpy.nonzero(alphas_soft > pred_nonconf)[0].tolist()) / len(alphas_soft)

    if true_ind == pred_ind:
        point_accuracy += 1

    if true_ind in region_inds_95:
        accuracy_95 += 1
    
    if true_ind in region_inds_90:
        accuracy_90 += 1
    
    if true_ind in region_inds_80:
        accuracy_80 += 1
    
    if true_ind in region_inds_75:
        accuracy_75 += 1

    true_inds.append(true_ind)

    if i % 100 == 0:
        print(i)
    
results = {}

lengths_95 = [len(i) for i in conf_intervals_soft_95]
lengths_90 = [len(i) for i in conf_intervals_soft_90]
lengths_80 = [len(i) for i in conf_intervals_soft_80]
lengths_75 = [len(i) for i in conf_intervals_soft_75]

results["lengths_95"] = [len(i) for i in conf_intervals_soft_95]
results["lengths_90"] = [len(i) for i in conf_intervals_soft_90]
results["lengths_80"] = [len(i) for i in conf_intervals_soft_80]
results["lengths_75"] = [len(i) for i in conf_intervals_soft_75]


results["mean_95"] = statistics.mean(lengths_95)
results["mean_90"] = statistics.mean(lengths_90)
results["mean_80"] = statistics.mean(lengths_80)
results["mean_75"] = statistics.mean(lengths_75)

results["med_95"] = statistics.median(lengths_95)
results["med_90"] = statistics.median(lengths_90)
results["med_80"] = statistics.median(lengths_80)
results["med_75"] = statistics.median(lengths_75)

emp_conf_95 = accuracy_95 / len(lengths_95)
emp_conf_90 = accuracy_90 / len(lengths_95)
emp_conf_80 = accuracy_80 / len(lengths_95)
emp_conf_75 = accuracy_75 / len(lengths_95)

results["emp_conf_95"] = emp_conf_95
results["emp_conf_90"] = emp_conf_90
results["emp_conf_80"] = emp_conf_80
results["emp_conf_75"] = emp_conf_75

print(emp_conf_95)
print(emp_conf_90)
print(emp_conf_80)
print(emp_conf_75)

results["classification_accuracy"] = point_accuracy / len(lengths_95)

results["OP"] = p_vals_true / len(lengths_95)
results["OF"] = p_vals_false / len(lengths_95)

results["credibility"] = creds / len(lengths_95)

#Save results to pickled file:
file_name = "results_" + seed
helper.save_obj(results, file_name)

import matplotlib.pyplot as plt
from matplotlib import colors
fig = plt
fig.xlabel("Non-conformity Scores")
fig.hist(alphas_soft, color = 'r')
#fig.show()
fig1_name = "Snapshot_Outs/Outfiles_mlm/non-confscores-plot" + "_" + seed + ".png"
fig.savefig(fig1_name)

y1 = [emp_conf_75, emp_conf_80, emp_conf_90, emp_conf_95]
x1 = [.75, .8, .9, .95]
x2 = [.75, .8, .9, .95]
y2 = [.75, .8, .9, .95]
fig2 = plt
fig2.plot(x1, y1, label = "Empirical")
fig2.plot(x2, y2, label = "Proposed")
fig2.xlim()
fig2.legend()
fig2.grid()
#fig2.show()
fig2_name = "Snapshot_Outs/Outfiles_mlm/empirical-proposed" + "_" + seed + ".png"
fig2.savefig(fig2_name)

fig3 = plt
fig3.hist(lengths_95, color='r')
#fig3.show()
fig3_name = "Snapshot_Outs/Outfiles_mlm/lengths-hist-95_" + seed + ".png"
