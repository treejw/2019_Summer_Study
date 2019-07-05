
from __future__ import print_function
from __future__ import division

import os
import torch
import pandas
import PIL.Image


class SOP_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, classes, transform = None):
        self.classes = classes
        self.root = root
        self.transform = transform
        self.ys, self.im_paths = [], []

        annos_fn = 'Ebay_info.txt'
        products = pandas.read_csv(os.path.join(root, annos_fn), sep=' ')
        ys_all = [(int(a)-1) for a in products['class_id']]
        im_paths_all = [a for a in products['path']]
        index = 0
        for im_path, y in zip(im_paths_all, ys_all):
            if y in classes: # choose only specified classes
                self.im_paths.append(os.path.join(root, im_path))
                self.ys.append(y)
                index += 1

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im = PIL.Image.open(self.im_paths[index])
        # convert gray to rgb
        if len(list(im.split())) == 1 : im = im.convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, self.ys[index], index

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

