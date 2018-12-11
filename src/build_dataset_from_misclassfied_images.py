from tqdm import tqdm
import os
from shutil import copyfile

# from targets import target_class_names

'''
make a dataset to validate the context-free models

run this after running find_misclassified_images
'''

from local_paths import *

if __name__ == '__main__':
    fname = "novel_image_paths.txt"

    f = open(fname, 'r')
    text = f.read()
    f.close()
    
    image_paths = text.strip().replace('\\','/').split('\n')

    for path in tqdm(image_paths):
        image_class, file_name = path.split('/')[-2:]
        class_path = '{}/val2017_unusual/{}/'.format(data_dir, image_class)
        os.makedirs(class_path, exist_ok=True)

        copyfile(path, class_path + file_name)
