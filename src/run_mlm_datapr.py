from os import getpid
import pandas as pd
import random
import sys, getopt
import routine_mlm as helper

    

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
    sentences = helper.load_sentencesdf()
    clean_sentences = helper.whole_sentences(sentences)
    
    #Generate masked indices
    mask_inds = helper.get_masked_inds(sentences)

    #Create new DF with cleaned sentences and masked inds
    new_df = pd.DataFrame([clean_sentences, mask_inds]).transpose()
    new_df.columns = ['clean_sentences',  'mask_inds']
    
    
    #Save new dfs with sentences and masked inds as tab seperated CSV
    new_df.to_csv("Data/brown_master.csv", sep = "\t", index = False)
     

if __name__ == "__main__":
    main(sys.argv[1:])


