import os
import numpy as np
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt
import glob
import scipy.misc

from targets import target_class_names

from local_paths import *

'''
change the backgrounds of the processed images

run this when you what to make a new background modification
'''



datatypes = "val2017 train2017".split()



def black_background(image, mask):
    output_image = image * mask[:,:,np.newaxis]


    # plt.imshow(output_image)
    # plt.axis('off')
    # plt.show()
    # exit()

    return output_image, "_black"


def random_background(image, mask):
    assert mask.max() == 1, mask.max()
    assert image.max() > 1, image.max() # expecting up to 255

    mask = mask[:,:,np.newaxis]

    noise = np.random.random(image.shape) * 255

    output_image = image * mask + (1 - mask) * noise

    # plt.imshow(output_image)
    # plt.axis('off')
    # plt.show()
    # exit()

    return output_image, "_random"


def only_background(image, mask):
    output_image = image * (1 - mask[:,:,np.newaxis])

    return output_image, "_only_background"

def alter_file(args):
    filename, dataType, target_class_name = args

    # cropped image
    cropped_image = io.imread(filename)
    if len(cropped_image.shape) < 3:
        cropped_image = np.repeat(cropped_image[:,:,np.newaxis], 3, axis=2)

    # cropped mask
    cropped_mask = io.imread(filename.replace('processed_images', 'processed_masks'))
    cropped_mask = (cropped_mask > 128).astype(int)

    # save the image
    output_filename = filename.replace('\\','/').split('/')[-1]
    
    output_image, folder_suffix = black_background(cropped_image, cropped_mask)
    # output_image, folder_suffix = random_background(cropped_image, cropped_mask)
    # output_image, folder_suffix = only_background(cropped_image, cropped_mask)

    # make the target folders
    output_dir = "{}/{}{}/{}".format(data_dir, dataType, folder_suffix, target_class_name)
    os.makedirs(output_dir, exist_ok=True)

    # print(output_image.shape)

    output_path = "{}/{}".format(output_dir, output_filename)
    scipy.misc.toimage(output_image, cmin=0.0, cmax=255.0).save(output_path)

if __name__ == '__main__':
    from multiprocessing import Pool
    pool = Pool(4)

    for dataType in datatypes:
        print(dataType)
        # dataDir = 'C:/Users/raffc/Downloads/coco2017'
        # annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
        processed_images_path = "{}/{}_processed_images".format(data_dir, dataType)

        for target_class_name in target_class_names:
            print("processing the '{}' class".format(target_class_name))

            image_paths = glob.glob('{}/{}/*'.format(processed_images_path, target_class_name))
            # mask_paths = glob.glob('{}/{}/*'.format(processed_masks_path, target_class_name))
            
            args = [(filename, dataType, target_class_name) for filename in image_paths]
            # image and mask filenames must be the same
            _ = list(tqdm(pool.imap(alter_file, args), total=len(image_paths)))

    '''
    load all the cropped images into RAM
    load all the cropped masks into RAM

    for each pair
        call the black_background function
        write to folder
    '''
