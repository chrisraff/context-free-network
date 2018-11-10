import numpy as np
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt
import glob
import scipy.misc

from targets import target_class_names


datatypes = "val2017 train2017".split()



def black_background(image, mask):
    output_image = image * mask[:,:,np.newaxis]


    # plt.imshow(output_image)
    # plt.axis('off')
    # plt.show()
    # exit()

    return output_image, "_black"






for dataType in datatypes:
    dataDir = '../res'
    # annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    processed_images_path = "{}/{}_processed_images".format(dataDir, dataType)
    processed_masks_path = "{}/{}_processed_masks".format(dataDir, dataType)

    for target_class_name in target_class_names:
        print("processing the '{}' class".format(target_class_name))

        image_paths = glob.glob('{}/{}/*'.format(processed_images_path, target_class_name))
        mask_paths = glob.glob('{}/{}/*'.format(processed_masks_path, target_class_name))

        print("loading images into RAM")
        images = []
        for filename in tqdm(image_paths):
            cropped_image = io.imread(filename)
            if len(cropped_image.shape) < 3:
                cropped_image = np.repeat(cropped_image[:,:,np.newaxis], 3, axis=2)
            images += [cropped_image]

        print("loading masks into RAM")
        masks = []
        for filename in tqdm(mask_paths):
            cropped_mask = io.imread(filename)
            cropped_mask = (cropped_mask > 128).astype(int)
            masks += [cropped_mask]

        # plt.imshow(images[0])
        # plt.axis('off')
        # plt.show()

        # plt.imshow(masks[0])
        # plt.axis('off')
        # plt.show()


        print("altering images")
        filenames = [x.split('/')[-1] for x in image_paths]
        for image, mask, filename in zip(images, masks, filenames):
            output_image, folder_suffix = black_background(image, mask)
            output_path = "{}/{}{}/{}".format(dataDir, dataType, folder_suffix, filename)
            scipy.misc.toimage(output_image, cmin=0.0, cmax=255.0).save(output_path)


'''
load all the cropped images into RAM
load all the cropped masks into RAM

for each pair
    call the black_background function
    write to folder
'''
