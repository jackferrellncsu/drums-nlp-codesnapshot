from os import getpid
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import sys, getopt
import routine_mlm as helper

raw_data = pd.read_csv("Data/brown.csv")
sentences = raw_data["raw_text"].str.split(expand = True)



def main(argv):
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
    
    print("Seed:", seed)
    #set seed from command line args
    random.seed(seed)

    #Clean sentences
    sentences, pos = helper.load_sentencesdf()
    clean_sentences = helper.whole_sentences(sentences)
    clean_pos = helper.whole_sentences(pos)
    
    #Generate masked indices
    mask_inds = helper.get_masked_inds(sentences)

    #Create new DF with cleaned sentences and masked inds
    new_df = pd.DataFrame([clean_sentences, clean_pos, mask_inds]).transpose()
    new_df.columns = ['clean_sentences', "clean_pos",  'mask_inds']
    
    train_val, test = train_test_split(new_df, train_size = 0.9)
    train, val  = train_test_split(train_val, train_size = 0.9)
    #Save new dfs with sentences and masked inds as tab seperated CSV
    new_df.to_csv("Data/brown_master.csv", sep = "\t", index = False)

    train.to_csv("Data/brown_train.csv", sep = "\t", index = False)
    test.to_csv("Data/brown_test.csv", sep = "\t", index = False)
    val.to_csv("Data/brown_validation.csv", sep = "\t", index = False)
     

if __name__ == "__main__":
    main(sys.argv[1:])


