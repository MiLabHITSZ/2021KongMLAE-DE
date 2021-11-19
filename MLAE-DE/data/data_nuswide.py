import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import pickle
import ast

tags = ['airport', 'animal', 'beach', 'bear', 'birds', 'boats', 'book', 'bridge', 'buildings', 'cars', 'castle', 'cat',
        'cityscape', 'clouds', 'computer', 'coral', 'cow', 'dancing', 'dog', 'earthquake', 'elk', 'fire', 'fish',
        'flags', 'flowers', 'food', 'fox', 'frost', 'garden', 'glacier', 'grass', 'harbor', 'horses', 'house', 'lake',
        'leaf', 'map', 'military', 'moon', 'mountain', 'nighttime', 'ocean', 'person', 'plane', 'plants', 'police',
        'protest', 'railroad', 'rainbow', 'reflection', 'road', 'rocks', 'running', 'sand', 'sign', 'sky', 'snow',
        'soccer', 'sports', 'statue', 'street', 'sun', 'sunset', 'surf', 'swimmers', 'tattoo', 'temple', 'tiger',
        'tower', 'town', 'toy', 'train', 'tree', 'valley', 'vehicle', 'water', 'waterfall', 'wedding', 'whales',
        'window', 'zebra']


def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


class NusWide(data.Dataset):
    def __init__(self, root, transform=None, phase='train', inp_name=None):
        self.root = root
        self.phase = phase
        self.transform = transform
        self.num_classes = 81
        self.get_anno()
        self.tags = tags
        if inp_name is not None:
            with open(inp_name, 'rb') as f:
                self.inp = pickle.load(f)
            self.inp_name = inp_name
        else:
            self.inp = None
            self.inp_name = None

    def get_anno(self):
        self.img_name_list = []
        self.tag_list = []
        img_list_path = os.path.join(self.root, 'nus_wide_data_{}.csv'.format(self.phase))
        with open(img_list_path, 'r') as f:
            reader = csv.reader(f)
            rownum = 0
            for row in reader:
                if rownum == 0:
                    pass
                else:
                    self.img_name_list.append(row[0].split('/')[1])
                    tag_names = ast.literal_eval(row[1])
                    tag = [-1 for i in range(self.num_classes)]
                    for tag_name in tag_names:
                        tag[tags.index(tag_name)] = 1
                    self.tag_list.append(tag)
                rownum += 1
    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, index):
        filename = self.img_name_list[index]
        tag = self.tag_list[index]
        img = Image.open(os.path.join(self.root, 'images', filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = np.asarray(tag)
        target[target==0] = -1

        if self.inp is None:
            return (img, filename), target
        else:
            return (img, filename, self.inp), target
