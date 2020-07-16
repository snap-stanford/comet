import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
import re

def read_img_id_pair(filename):
    img_to_id = dict()
    with open(filename, 'r') as fin:
        for line in fin.readlines():
            line_split = line.strip().split(' ')
            img_to_id[line_split[1]] = int(line_split[0])
    return img_to_id

def read_parts(filename):
    id_to_parts = dict()
    with open(filename, 'r') as fin:
        for line in fin.readlines():
            line_split = line.strip().split(' ')
            img_id, part_id, x, y, visible = int(line_split[0]), int(line_split[1]), float(line_split[2]), float(line_split[3]), int(line_split[4])
            if part_id == 1:
                id_to_parts[img_id] = [[x, y, visible], ]
            else:
                id_to_parts[img_id].append([x, y, visible])
    return id_to_parts

if __name__ == '__main__':

    img_to_idx = read_img_id_pair('./CUB_200_2011/images.txt')
    id_to_parts = read_parts('./CUB_200_2011/parts/part_locs.txt')

    cwd = os.getcwd() 
    data_path = join(cwd,'CUB_200_2011/images')
    savedir = './'
    dataset_list = ['base','val','novel']

    folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
    folder_list.sort()
    label_dict = dict(zip(folder_list,range(0,len(folder_list))))

    classfile_list_all = []

    for i, folder in enumerate(folder_list):
        folder_path = join(data_path, folder)
        classfile_list_all.append( [ join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')])
        random.shuffle(classfile_list_all[i])


    for dataset in dataset_list:
        file_list = []
        label_list = []
        for i, classfile_list in enumerate(classfile_list_all):
            if 'base' in dataset:
                if (i%2 == 0):
                    file_list = file_list + classfile_list
                    label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
            if 'val' in dataset:
                if (i%4 == 1):
                    file_list = file_list + classfile_list
                    label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
            if 'novel' in dataset:
                if (i%4 == 3):
                    file_list = file_list + classfile_list
                    label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        part_list = []
        for path in file_list:        
            filename = re.search('/images/(.*)', path, flags=0).group(1)
            part_list.append(id_to_parts[img_to_idx[filename]])
        with open(savedir + dataset + '.json', 'w') as outfile:
            json.dump({'label_names':folder_list, 'image_names':file_list, 'image_labels':label_list, 'part': part_list}, outfile)

        print("%s -OK" %dataset)
