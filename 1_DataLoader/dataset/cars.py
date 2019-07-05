
from __future__ import print_function
from __future__ import division

import scipy.io
import os
import torch
import PIL.Image


class Cars_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, classes, transform = None):
        self.classes = classes
        self.root = root
        self.transform = transform
        self.ys, self.im_paths = [], []

        annos_fn = 'cars_annos.mat'
        cars = scipy.io.loadmat(os.path.join(root, annos_fn))
        ys_all = [int(a[5][0] - 1) for a in cars['annotations'][0]]
        im_paths_all = [a[0][0] for a in cars['annotations'][0]]
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
