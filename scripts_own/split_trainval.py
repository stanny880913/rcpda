'''
Split data with moving ego vehicles on a clear day.
'''

import os
import numpy as np
import torch
import argparse
import cv2
import re
from PIL import Image

from nuscenes.nuscenes import NuScenes

np.random.seed(1)

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group(0)) if match else -1

def read_jpg_images_from_folder(folder_path):
    image_list = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and is_jpg_image_file(filename):
            try:
                image = cv2.imread(file_path)
                if image is not None:
                    image_list.append(image)
                else:
                    print(f"Unable to read image file: {file_path}")
            except Exception as e:
                print(f"Failed to read image file: {file_path}, Error: {str(e)}")
    return image_list

def is_jpg_image_file(filename):
    ext = os.path.splitext(filename)[-1].lower()
    return ext == '.jpg'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', default='/media/stannyho/ssd/rc-pda/data_own',type=str)
    # parser.add_argument('--version', type=str, default='v1.0-trainval')
    args = parser.parse_args()

    dir_own = '/media/stannyho/ssd/rc-pda/data_own/img_jpg'

    own_data = read_jpg_images_from_folder(dir_own)
    print(len(own_data), 'own_data len')
    own_file_names = os.listdir(dir_own)
    print(len(own_file_names),'files name len')
    own_file_names.sort(key=extract_number)
    own_sample_files = own_file_names

    all_idx = own_data

    data_split = {'all_indices': all_idx,
                  'own_sample_files': own_sample_files
                 }

    torch.save(data_split, os.path.join(args.dir_data, 'own_data_split.tar'))
