from __future__ import print_function
from __future__ import division

import torch
import PIL.Image
import torchvision
import os


class CUB_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, classes, transform = None):
        self.classes = classes
        self.root = root
        self.transform = transform
        self.ys, self.im_paths = [], []

        index = 0
        for i in torchvision.datasets.ImageFolder(root=os.path.join(root, 'images')).imgs:
            # i[0]: root, i[1]: label
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.classes and fn[:2] != '._':
                self.ys += [y]
                self.im_paths.append(os.path.join(root, i[0]))
                index += 1

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im = PIL.Image.open(self.im_paths[index])
        # convert gray to rgb
        if len(list(im.split())) == 1 : 
            im = im.convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, self.ys[index], index

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)



if __name__ == '__main__':

    from utils import make_transform

    Dataset = CUB_Dataset(root='/home/artint/다운로드/CUB_200_2011',
                          classes=range(0,100),
                          transform=make_transform(rgb_to_bgr = True,
                                                   is_train=True)
                          )

    DataLoader = torch.utils.data.DataLoader(dataset=Dataset,
                                             batch_size=4,
                                             shuffle=True
                                             )

    f = open('test.txt', 'w', encoding='utf-8')

    for epoch in range(1):
        for i, data in enumerate(DataLoader):
            inputs, labels, idxs = data
            print('\nbatch : {} / \ninputs : {}\nlabels : {}\nidxs : {}'.format(i, inputs.size(), labels, idxs))
            f.write('\n\nbatch : {} / \ninputs : {}\nlabels : {}\nidxs : {}'.format(i, inputs.size(), labels, idxs))

    f.close()
