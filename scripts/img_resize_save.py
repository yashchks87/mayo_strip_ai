import numpy as np
from PIL import Image
import multiprocessing as mp
from itertools import repeat
from tqdm import tqdm
import os
import argparse 
import pandas as pd
import PIL

def img_resize_save_helper(data: list) -> str:
    try:
        img_path, size, save_location = data
        img_name = img_path.split('/')[-1].split('.')[0]
        if os.path.exists(save_location + img_name + '.jpg') == False:
            img = Image.open(img_path)
            img = img.resize(size)
            img.save(save_location + img_name + '.jpg')
            return 'True'
        else:
            return 'True'
    except:
        return 'False'

def img_resize(img_paths: list, img_size: tuple, save_location: str, mp_yes: bool, pool_size : int) -> list:
    if os.path.exists(save_location) == False:
        os.makedirs(save_location)
    data_loader = list(zip(img_paths, repeat(img_size), repeat(save_location)))
    if mp_yes:
        with mp.Pool(pool_size) as p:
            returns = list(p.map(img_resize_save_helper, data_loader))
    else:
        returns = []
        for x in tqdm(data_loader):
            returns.append(img_resize_save_helper(x))
    issues = []
    for x in tqdm(range(len(returns))):
        if returns[x] == 'False':
            issues.append(img_paths[x])
    return issues
    
if __name__ == '__main__':
    Image.MAX_IMAGE_PIXELS = None
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv_file')
    parser.add_argument('-is', '--img_size', type=int)
    parser.add_argument('-s', '--save_location')
    parser.add_argument('-mp', '--mp_yes', default=False, type=bool)
    parser.add_argument('-p', '--pool_size', default=2, type = int)
    parser.add_argument('-b', '--basic_location', default='../../mayo_data/train/')
    args = parser.parse_args()
    assert os.path.exists(args.csv_file) == True, 'csv file does not exist.'
    cf = pd.read_csv(args.csv_file)
    cf['updated_paths'] = cf['image_id'].apply(lambda x : args.basic_location + x + '.tif')
    paths = cf['updated_paths'].values.tolist()
    img_size = (args.img_size, args.img_size)
    returns = img_resize(paths, img_size, args.save_location, args.mp_yes, args.pool_size)
    print(returns)

# Demo string
# python img_resize_save.py -c ../../mayo_data/train.csv -is 2048 -s ../../mayo_data/train_resized/2048/