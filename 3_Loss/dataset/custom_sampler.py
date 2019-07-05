from __future__ import absolute_import

from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    def __init__(self, dataset, num_identities=60, num_instances=3):
        self.num_data = len(dataset)
        self.num_identities = num_identities
        self.num_instances = num_instances

        self.index_dic = defaultdict(list)
        for index, label in enumerate(dataset.ys):
            self.index_dic[label].append(index)
        self.labels = list(self.index_dic.keys())

    def __iter__(self):
        self.ret = []

        while len(self.ret) <= self.num_data - (self.num_identities * self.num_instances):
            replace_identities = False if len(self.labels) >= self.num_identities else True
            indices = np.random.choice(self.labels, size=self.num_identities, replace=replace_identities)
            for i in indices:
                label = self.labels[i]
                t = self.index_dic[label]
                replace_instance = False if len(t) >= self.num_instances else True
                t = np.random.choice(t, size=self.num_instances, replace=replace_instance)
                self.ret.extend(t)
        return iter(self.ret)

    def __len__(self):
        return len(self.ret)
