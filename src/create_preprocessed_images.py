from pycocotools.coco import COCO
from pycocotools import mask as maskutils
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import scipy
from skimage.transform import resize as imresize
from time import sleep
import scipy.misc
from tqdm import tqdm
import os

from targets import target_class_names

from local_paths import *

'''
compute the crops and masks for the images

run this if you want to introduce new classes
'''


datatypes = "val2017 train2017".split()

for dataType in datatypes:
    # data_dir = 'C:/Users/raffc/Downloads/coco2017'
    # dataType = 'val2017'
    annFile = '{}/annotations/instances_{}.json'.format(data_dir, dataType)
    processed_images_path = "{}/{}_processed_images".format(data_dir, dataType)
    processed_masks_path = "{}/{}_processed_masks".format(data_dir, dataType)

    # initialize COCO api for instance annotations
    coco = COCO(annFile)

    # display COCO categories and supercategories
    categories = coco.loadCats(coco.getCatIds())
    category_name_dict = {}
    for category in categories:
        if category['name'] not in target_class_names: continue
        category_name_dict[category['id']] = category['name']

        # make folders for classes
        for output_dir in [processed_images_path, processed_masks_path]:
            os.makedirs('{}/{}'.format(output_dir, category['name']), exist_ok=True)
    # category_names = [cat['name'] for cat in categories]
    # print('COCO categories: \n{}\n'.format(' '.join(category_names)))



    '''
    for each dataset
        for each image in the data set
            for each bounding box in the image for a class that we care about
                crop to that bounding box
                remove the background

                save the cropped image
                save the background mask
                    (make sure to nicely encode the class and image id when we do that)
    '''





    target_class_ids = coco.getCatIds(catNms=target_class_names)
    target_image_ids = set()  # set of all image ids that have any class that we care about
    for target_class_id in target_class_ids:
        imgIds = coco.getImgIds(catIds=target_class_id)
        target_image_ids.update(imgIds)  # add the image ids to the set
    target_image_ids = list(target_image_ids)






    def process_image(image_id):
        '''
        get the image pixels
        get the segmentation mask
        crop it
        save the image to processed_images
        save the mask to processed_masks
        '''

        img = coco.loadImgs(image_id)[0]
        pixels = io.imread("{}/{}/{}".format(data_dir, dataType, img['file_name']))



        annIds = coco.getAnnIds(imgIds=img['id'], catIds=coco.getCatIds(), iscrowd=False)
        anns = coco.loadAnns(annIds)
        # for each json blob (bounding box annotation)
        for obj in anns:
            if obj['category_id'] not in target_class_ids:
                continue

            # get the bounding box
            y,x,h,w = [int(num) for num in obj['bbox']]

            # don't take super small images
            if h < 32 or w < 32:
                continue


            # crop the image
            cropped_image = pixels[x:x+w, y:y+h]

            # # show the image
            # plt.imshow(cropped_image)
            # plt.axis('off')
            # plt.show()

            # get the mask
            rle = maskutils.frPyObjects(obj['segmentation'], *pixels.shape[:2])
            mask = maskutils.decode(rle)
            cropped_mask = mask[x:x+w, y:y+h,0]

            object_coverage = np.sum(mask) / np.prod(cropped_mask.shape)

            # ignore images that take up too little or too much of the cropped image
            if not 0.1 < object_coverage < 0.95:
                continue

            # normalize the width and height of the image and the mask
            output_size = 224
            final_img = imresize(cropped_image, (output_size,output_size))
            final_mask = imresize(cropped_mask, (output_size,output_size), order=0, preserve_range=True)

            # # the image with the background zeroed out using the mask (useful for later)
            # alignment_test_img = final_img * final_mask[:,:,np.newaxis]
            # plt.imshow(alignment_test_img)
            # plt.show()


            # save the image
            category_id = obj['category_id']
            image_name = "{}_{}_{}".format(category_id, image_id, obj['id'])
            image_path = "{}/{}/{}.jpg".format(processed_images_path, category_name_dict[category_id], image_name)
            scipy.misc.toimage(final_img, cmin=0.0, cmax=1.0).save(image_path)

            # save the mask
            mask_path = "{}/{}/{}.jpg".format(processed_masks_path, category_name_dict[category_id], image_name)
            scipy.misc.toimage(final_mask, cmin=0.0, cmax=1.0).save(mask_path)

        return


    # DEBUG: only using the validation set currently
    # process all the images
    # for target_image_id in  (target_image_ids):
    for target_image_id in tqdm(target_image_ids):
        process_image(target_image_id)
