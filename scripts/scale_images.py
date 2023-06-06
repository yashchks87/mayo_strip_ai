# Import libraries
import numpy as np
import PIL
from PIL import Image
import glob
import multiprocessing as mp
import os
import argparse
# Needed because of issue of image sizes are too big
PIL.Image.MAX_IMAGE_PIXELS = 21048338080
import pickle

def get_files(dir_path):
    # This function get total images from the given dir path
    files = glob.glob(f'{dir_path}*.tif')
    return files

def resize_images(data):
    # This is running as multiprocessing function
    img_path, updated_path, size = data[0], data[1], data[2]
    file_name = img_path.split('/')[-1].split('.')[0]
    try:
        img = PIL.Image.open(img_path)
        img = img.resize((size, size))
        img.save(f'{updated_path}{file_name}.png')
        return 'DONE'
    except:
        return img_path
    
def scale_images_and_store(dir_path, store_path, size, pickle_location):
    files = get_files(dir_path)
    if os.path.exists(store_path) == False:
        os.makedirs(store_path)
    updated_data = [[files[x], store_path, size] for x in range(len(files))]
    with mp.Pool(8) as p:
        results = list(p.map(resize_images, updated_data))
    issue_images = []
    for x in results:
        if x != 'DONE':
            issue_images.append(x)
    if len(issue_images) > 0:
        with open(pickle_location, 'wb') as handle:
            pickle.dump(issue_images, handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog='scale_images',
                        description='Scale images and store them',
                        epilog='Text at the bottom of help')

    parser.add_argument('-d', '--dirpath')
    parser.add_argument('-s', '--storepath')
    parser.add_argument('-i', '--image_size')
    parser.add_argument('-p', '--pickle_location')

    args = parser.parse_args()

    dir_path = args.dirpath
    store_path = args.storepath
    image_size = int(args.image_size)
    pickle_location = args.pickle_location

    scale_images_and_store(dir_path, store_path, image_size, pickle_location)

# Please go inside the scripts folder and execcute this script
# python scale_images.py -d ../../files/train/ -s ../../files/resized_train/ -i 256 -p ../issue_images.pickle