import pickle
import statistics
import math
import routine_mlm as helper
import os
import sys, getopt

#get seed arg from command line here
seed = helper.command_line_seed(sys.argv[1:])
result_objs = []

for i in range(5):
    with open(f"out/out_mlm_results/results_{seed}_{i}.pkl", "rb") as file:
        
        result = pickle.load(file)
    result_objs.append(result)

lengthst_95 = []
lengthst_90 = []
lengthst_80 = []
lengthst_75 = []

means_95 = []
means_90 = []
means_80 = []
means_75 = []

emp_confs_95 = []
emp_confs_90 = []
emp_confs_80 = []
emp_confs_75 = []

cas = []
ops = []
ofs = []
creds = []


for i in range(len(result_objs)):
    lengthst_95 = lengthst_95 + result_objs[i]['lengths_95']
    lengthst_90 = lengthst_90 + result_objs[i]['lengths_90']
    lengthst_80 = lengthst_80 + result_objs[i]['lengths_80']
    lengthst_75 = lengthst_75 + result_objs[i]['lengths_75']

    means_95.append(result_objs[i]['mean_95'])
    means_90.append(result_objs[i]['mean_90'])
    means_80.append(result_objs[i]['mean_80'])
    means_75.append(result_objs[i]['mean_75'])

    emp_confs_95.append(result_objs[i]['emp_conf_95'])
    emp_confs_90.append(result_objs[i]['emp_conf_90'])
    emp_confs_80.append(result_objs[i]['emp_conf_80'])
    emp_confs_75.append(result_objs[i]['emp_conf_75'])

    cas.append(result_objs[i]['classification_accuracy'])
    ops.append(result_objs[i]['OP'])
    ofs.append(result_objs[i]['OF'])
    creds.append(result_objs[i]['credibility'])

#Reportable results:
with open(f"out/out_mlm_results/result_report_{seed}.txt", "w") as file:

    #Median:
    med_tot_95 = statistics.median(lengthst_95)
    med_tot_90 = statistics.median(lengthst_90)
    med_tot_80 = statistics.median(lengthst_80)
    med_tot_75 = statistics.median(lengthst_75)

    print(f'Median Set Size 95%: {med_tot_95}', file = file)
    print(f'Median Set Size 90%: {med_tot_90}', file = file)
    print(f'Median Set Size 80%: {med_tot_80}', file = file)
    print(f'Median Set Size 75%: {med_tot_75}', file = file)

    #Mean:
    mean_tot_95 = statistics.mean(means_95)
    mean_tot_90 = statistics.mean(means_90)
    mean_tot_80 = statistics.mean(means_80)
    mean_tot_75 = statistics.mean(means_75)

    print(f'Mean Set Size 95%: {mean_tot_95}', file=file)
    print(f'Mean Set Size 90%: {mean_tot_90}', file=file)
    print(f'Mean Set Size 80%: {mean_tot_80}', file=file)
    print(f'Mean Set Size 75%: {mean_tot_75}', file=file)

    #Confidences:
    conf_tot_95 = statistics.mean(emp_confs_95)
    conf_tot_90 = statistics.mean(emp_confs_90)
    conf_tot_80 = statistics.mean(emp_confs_80)
    conf_tot_75 = statistics.mean(emp_confs_75)

    print(f'Average Empirical Confidence 95%: {conf_tot_95}', file=file)
    print(f'Average Empirical Confidence 90%: {conf_tot_90}', file=file)
    print(f'Average Empirical Confidence 80%: {conf_tot_80}', file=file)
    print(f'Average Empirical Confidence 75%: {conf_tot_75}', file=file)

    #OPs:
    op_tot = statistics.mean(ops)
    print(f'Average OP Score: {op_tot}', file=file)

    #OFs:
    of_tot = statistics.mean(ofs)
    print(f'Average OF Score: {of_tot}', file=file)

    #CA:
    ca_tot = statistics.mean(cas)
    print(f'Average Classification Accuracy: {ca_tot}', file=file)

    #Cred:
    cred_tot = statistics.mean(creds)
    print(f'Average Credibility: {cred_tot}', file=file)

#First saved alphas from the 5 runs will be loaded:
alpha_filename = f"soft_alphas_{seed}_0.txt"
alphas = helper.load_cal_alphas(alpha_filename)


import matplotlib.pyplot as plt
from matplotlib import colors
fig = plt
bin_num = math.ceil(math.sqrt(len(alphas)))
fig.hist(alphas, bins = bin_num,  color = 'r')
#fig.show()
fig1_name = f"out/out_mlm_results/non-confscores-plot_{seed}.png"
fig.savefig(fig1_name)
fig.close()

fig4 = plt
alphas.sort()
alpha_zoom = filter(lambda x: x>.9975, alphas)
print(alphas[-60:])
#hist of last ~300 elements of sorted list
bin_num = math.ceil(math.sqrt(len(alphas[-60:])))
fig4.hist(alphas[-60:], bins = bin_num, color='r')
fig4_name = f"out/out_mlm_results/non-confscores-plot_zoom_{seed}.png"
fig4.savefig(fig4_name)
fig4.close()

y1 = [conf_tot_75, conf_tot_80, conf_tot_90, conf_tot_95]
x1 = [.75, .8, .9, .95]
x2 = [.75, .8, .9, .95]
y2 = [.75, .8, .9, .95]
xi = list(range(len(x1)))
fig2 = plt
fig2.plot(x1, y1, label = "Empirical")
fig2.plot(x2, y2, label = "Nominal")
fig2.legend()
fig2_name = f"out/out_mlm_results/empirical-proposed_{seed}.png"
fig2.savefig(fig2_name)
fig2.close()

fig3 = plt
bin_num = math.ceil(math.sqrt(len(lengthst_95)))
fig3.hist(lengthst_95, bins=bin_num, color='r')
#fig3.show()
fig3_name = f"out/out_mlm_results/lengths-hist-95_{seed}.png"
fig3.savefig(fig3_name)
fig3.close()
