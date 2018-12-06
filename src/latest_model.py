from os import listdir
from os.path import isfile, join


def latest_model():
    models_path = "../models"
    model_fnames = [f for f in listdir(models_path) if isfile(join(models_path, f))]
    model_fnames = sorted(model_fnames, reverse=True)
    latest_model_fname = model_fnames[0]
    return latest_model_fname
