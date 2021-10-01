import json

#loads a list from .txt file in json format
def load_cal_alphas(filename):
    path = "Data/" + filename + '.txt'
    with open(path, "r") as file:
        loaded_vals = json.load(file)
    return loaded_vals