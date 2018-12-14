from multiprocessing import Pool, cpu_count
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



def process_image(args):
    image_id, coco, dataType, target_class_ids, processed_images_path, category_name_dict = args
    '''
    get the image pixels
    get the segmentation mask
    crop it
    save the image to processed_images
    save the mask to processed_masks
    '''

    img = coco.loadImgs(image_id)[0]
    original_pixels = io.imread("{}/{}/{}".format(data_dir, dataType, img['file_name']))



    annIds = coco.getAnnIds(imgIds=img['id'], catIds=coco.getCatIds(), iscrowd=False)
    anns = coco.loadAnns(annIds)
    # for each json blob (bounding box annotation)
    for obj in anns:
        pixels = np.copy(original_pixels)

        if obj['category_id'] not in target_class_ids:
            continue

        # get the bounding box
        y,x,h,w = [int(num) for num in obj['bbox']]

        # don't take super small images
        if h < 32 or w < 32:
            continue

        # # show the image
        # plt.imshow(pixels)
        # plt.axis('off')
        # plt.show()


        # black out the object
        pixels[x:x+w, y:y+h] = 0


        # crop the image to the area around the bounding box
        border_proportion = 1/3  # percentage of the bbox width to pad the crop with

        start_x = int(max(0, x - w*border_proportion))
        end_x = int(min(pixels.shape[0], start_x + w + 2*w*border_proportion))
        start_y = int(max(0, y - h*border_proportion))
        end_y = int(min(pixels.shape[1], start_y + h + 2*h*border_proportion))
        cropped_image = pixels[start_x:end_x, start_y:end_y]


        # # show the image
        # plt.subplot(2, 1, 2)
        # plt.imshow(cropped_image)
        # plt.axis('off')
        # plt.show()



        # ignore images with 1 class that fills up the whole image
        total_image_object_coverage = (w*h) / (pixels.shape[0]*pixels.shape[1])
        if total_image_object_coverage > 0.9:
            # print("skipping 1")
            continue

        # ignore images that take up too little or too much of the cropped image
        cropped_image_object_coverage = (w*h) / (cropped_image.shape[0]*cropped_image.shape[1])
        if not (0.1 < cropped_image_object_coverage < 0.9):
            # print("skipping 2")
            continue

        # # created the cropped mask
        # # cropped_mask = np.zeros_like(cropped_image)
        # start_x = int(w*border_proportion)
        # end_x = int(start_x + w)
        # start_y = int(h*border_proportion)
        # end_y = int(start_y + h)
        # # cropped_mask[start_x:end_x, start_y:end_y] = 1

        # cropped_image[start_x:end_x, start_y:end_y] = 0

        # # get the mask
        # rle = maskutils.frPyObjects(obj['segmentation'], *pixels.shape[:2])
        # mask = maskutils.decode(rle)
        # cropped_mask = mask[x:x+w, y:y+h,0]


        # object_coverage = np.sum(mask) / np.prod(cropped_mask.shape)

        # # ignore images that take up too little or too much of the cropped image
        # if not 0.1 < object_coverage < 0.9:
        #     continue


        # normalize the width and height of the image and the mask
        output_size = 224
        final_img = imresize(cropped_image, (output_size,output_size), mode='reflect')
        # final_mask = imresize(cropped_mask, (output_size,output_size), order=0, preserve_range=True)


        # # the image with the background zeroed out using the mask (useful for later)
        # alignment_test_img = final_img * final_mask[:,:,np.newaxis]
        # plt.imshow(alignment_test_img)
        # plt.show()


        # save the image
        category_id = obj['category_id']
        image_name = "{}_{}_{}".format(category_id, image_id, obj['id'])
        image_path = "{}/{}/{}.jpg".format(processed_images_path, category_name_dict[category_id], image_name)
        scipy.misc.toimage(final_img, cmin=0.0, cmax=1.0).save(image_path)

        # # save the mask
        # mask_path = "{}/{}/{}.jpg".format(processed_masks_path, category_name_dict[category_id], image_name)
        # scipy.misc.toimage(final_mask, cmin=0.0, cmax=1.0).save(mask_path)


    return



if __name__ == '__main__':
    threaded = False
    pool = Pool(cpu_count()//2)
    # pool = Pool(cpu_count())

    datatypes = "val2017 train2017".split()

    for dataType in datatypes:
        print("processing {}".format(dataType))
        # data_dir = 'C:/Users/raffc/Downloads/coco2017'
        # dataType = 'val2017'
        annFile = '{}/annotations/instances_{}.json'.format(data_dir, dataType)
        processed_images_path = "{}/{}_context_only".format(data_dir, dataType)
        # processed_masks_path = "{}/{}_processed_masks".format(data_dir, dataType)

        # initialize COCO api for instance annotations
        coco = COCO(annFile)

        # display COCO categories and supercategories
        categories = coco.loadCats(coco.getCatIds())
        category_name_dict = {}
        for category in categories:
            if category['name'] not in target_class_names:
                continue
            category_name_dict[category['id']] = category['name']

            # make folders for classes
            for output_dir in [processed_images_path]:
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



        # DEBUG: only using the validation set currently
        # process all the images
        # for target_image_id in  (target_image_ids):
        # (process_image(target_image_id) for target_image_id in tqdm(target_image_ids))

        args0 = target_image_ids
        args1 = [coco]*len(target_image_ids)
        args2 = [dataType]*len(target_image_ids)
        args3 = [target_class_ids]*len(target_image_ids)
        args4 = [processed_images_path]*len(target_image_ids)
        args5 = [category_name_dict]*len(target_image_ids)
        args = list(zip(args0, args1, args2, args3, args4, args5))

        if threaded:
            _ = list(tqdm(pool.imap(process_image, args), total=len(target_image_ids)))
        else:
            _ = list(tqdm(map(process_image, args), total=len(target_image_ids)))
