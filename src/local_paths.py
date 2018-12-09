from os import listdir
from os.path import isfile, join


models_dir = "../models"

data_dir = '../res'
# data_dir = 'C:/Users/raffc/Downloads/coco2017'

templates_dir = data_dir+'/templates'


def latest_model(mode="random"):
    model_fnames = [f for f in listdir(models_dir) if isfile(join(models_dir, f))]
    model_fnames = [fname for fname in model_fnames if mode in fname]
    model_fnames = sorted(model_fnames, reverse=True)
    latest_model_fname = model_fnames[0]
    return models_dir+'/'+latest_model_fname
