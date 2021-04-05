from torch.utils.data.sampler import Sampler
import random
import torch.distributed as dist
import numpy as np
import math
import json


class DistributedCombinedRandomSampler(Sampler):
    def __init__(self, num_weakly, num_full, batch_size, ratio_weakly, num_replicas=None, rank=None, overlap_file=None):
        # super(CombinedRandomSampler, self).__init__()
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # self.num_samples = int(math.ceil((num_weakly / (ratio_weakly) / self.num_replicas)))
        self.num_samples = int(math.ceil((num_full / (1 - ratio_weakly) / self.num_replicas)))
        self.total_size = self.num_samples * self.num_replicas

        assert batch_size >= 2, 'batch_size must be larger than 1, get {}'.format(batch_size)
        assert ratio_weakly < 1, 'ratio_weakly must be smaller than 1, get {}'.format(ratio_weakly)
        self.num_weakly = num_weakly
        self.num_full = num_full
        self.batch_size = batch_size
        self.weakly_in_batch = max(int(batch_size*ratio_weakly), 1)
        self.full_in_batch = batch_size - self.weakly_in_batch
        
        if overlap_file:
            with open(overlap_file, 'r') as fo:
                self.overlap_anno_id = list(set(json.load(fo)))
                print('successfully load ids of large overlap instances!')
        else:
            self.overlap_anno_id = None
            

    # # weakly datas as major datas
    # def __iter__(self):
    #     weakly_ls = list(range(self.num_weakly))
    #     full_ls = list(range(self.num_weakly, self.num_weakly + self.num_full))
    #     random.shuffle(weakly_ls)
    #     random.shuffle(full_ls)
    #
    #     weakly_inds = 0
    #     full_inds = 0
    #     batch_ls = []
    #     while (weakly_inds + self.weakly_in_batch) <= self.num_weakly:
    #         if (full_inds + self.full_in_batch) <= self.num_full:
    #             batch = weakly_ls[weakly_inds: weakly_inds + self.weakly_in_batch] + \
    #                     full_ls[full_inds: full_inds + self.full_in_batch]
    #             random.shuffle(batch)
    #             batch_ls.append(batch)
    #             weakly_inds += self.weakly_in_batch
    #             full_inds += self.full_in_batch
    #         else:
    #             batch = weakly_ls[weakly_inds: weakly_inds + self.weakly_in_batch] + \
    #                     full_ls[full_inds:] + full_ls[0: full_inds + self.full_in_batch - self.num_full]
    #             random.shuffle(batch)
    #             batch_ls.append(batch)
    #             weakly_inds += self.weakly_in_batch
    #             full_inds = full_inds + self.full_in_batch - self.num_full
    #
    #     if weakly_inds < self.num_weakly and (full_inds+self.batch_size-self.num_weakly+weakly_inds) <= self.num_full:
    #         batch = weakly_ls[weakly_inds:] + full_ls[full_inds:full_inds+self.batch_size-self.num_weakly+weakly_inds]
    #         random.shuffle(batch)
    #         batch_ls.append(batch)
    #     if weakly_inds < self.num_weakly and (full_inds+self.batch_size-self.num_weakly+weakly_inds) > self.num_full:
    #         batch = weakly_ls[weakly_inds:] + full_ls[full_inds:] + \
    #                 full_ls[0: full_inds+self.batch_size-self.num_weakly+weakly_inds - self.num_full]
    #         random.shuffle(batch)
    #         batch_ls.append(batch)
    #
    #     indices = [item for batch in batch_ls for item in batch]
    #
    #     # add extra samples to make it evenly divisible
    #     indices += indices[:(self.total_size - len(indices))]
    #     assert len(indices) == self.total_size
    #
    #     # subsample
    #     indices = indices[self.rank:len(indices):self.num_replicas]
    #     assert len(indices) == self.num_samples
    #
    #     return iter(indices)

    # full supervised datas as major datas
    def __iter__(self):
        weakly_ls = list(range(self.num_weakly))
        if self.overlap_anno_id:
            # random drop overlap instances
            for idx in self.overlap_anno_id:
                if random.random()<=0.75:
                    try:
                        weakly_ls.remove(idx)
                    except Exception:
                        pass
            if self.rank == 0:
                print('Sample {} weakly-supervised instances in this epoch.'.format(len(weakly_ls)))
        
        full_ls = list(range(self.num_weakly, self.num_weakly + self.num_full))
        random.shuffle(weakly_ls)
        random.shuffle(full_ls)
        
        weakly_inds = 0
        full_inds = 0
        batch_ls = []
        len_weakly_ls = len(weakly_ls)
        while (full_inds + self.full_in_batch) <= self.num_full:
            if (weakly_inds + self.weakly_in_batch) <= len_weakly_ls :
                batch = full_ls[full_inds: full_inds + self.full_in_batch] + \
                        weakly_ls[weakly_inds: weakly_inds + self.weakly_in_batch]
                random.shuffle(batch)
                batch_ls.append(batch)
                weakly_inds += self.weakly_in_batch
                full_inds += self.full_in_batch
            else:
                batch = full_ls[full_inds: full_inds + self.full_in_batch] + \
                        weakly_ls[weakly_inds:] + weakly_ls[0: weakly_inds + self.weakly_in_batch - len_weakly_ls]
                random.shuffle(batch)
                batch_ls.append(batch)
                weakly_inds += self.weakly_in_batch
                full_inds = full_inds + self.full_in_batch - self.num_full

        if full_inds < self.num_full and (weakly_inds+self.batch_size-self.num_full+full_inds) <= len_weakly_ls:
            batch = full_ls[full_inds:] + weakly_ls[weakly_inds:weakly_inds+self.batch_size-self.num_full+full_inds]
            random.shuffle(batch)
            batch_ls.append(batch)
        if full_inds < self.num_full and (full_inds+self.batch_size-self.num_full+weakly_inds) > len_weakly_ls:
            batch = full_ls[full_inds:] + weakly_ls[weakly_inds:] + \
                    weakly_ls[0: weakly_inds+self.batch_size-self.num_full+weakly_full - len_weakly_ls]
            random.shuffle(batch)
            batch_ls.append(batch)

        indices = [item for batch in batch_ls for item in batch]

        # add extra samples to make it evenly divisible
        # print(self.num_weakly, self.num_full, len(indices), self.total_size)
        if len(indices)<self.total_size:
            indices += indices[:(self.total_size - len(indices))]
        else:
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:len(indices):self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class CombinedRandomSampler(Sampler):
    def __init__(self, num_weakly, num_full, batch_size, ratio_weakly, overlap_file=None):
        # super(CombinedRandomSampler, self).__init__()
        assert batch_size >= 2, 'batch_size must be larger than 1, get {}'.format(batch_size)
        assert ratio_weakly < 1, 'ratio_weakly must be smaller than 1, get {}'.format(ratio_weakly)
        self.num_weakly = num_weakly
        self.num_full = num_full
        self.batch_size = batch_size
        self.weakly_in_batch = max(int(batch_size*ratio_weakly), 1)
        self.full_in_batch = batch_size - self.weakly_in_batch

        if overlap_file:
            with open(overlap_file, 'r') as fo:
                self.overlap_anno_id = list(set(json.load(fo)))
                print('successfully load ids of large overlap instances!')
        else:
            self.overlap_anno_id = None

    # # weakly data as major data
    # def __iter__(self):
    #     weakly_ls = list(range(self.num_weakly))
    #     full_ls = list(range(self.num_weakly, self.num_weakly + self.num_full))
    #     random.shuffle(weakly_ls)
    #     random.shuffle(full_ls)
    #
    #     weakly_inds = 0
    #     full_inds = 0
    #     batch_ls = []
    #     while (weakly_inds + self.weakly_in_batch) <= self.num_weakly:
    #         if (full_inds + self.full_in_batch) <= self.num_full:
    #             batch = weakly_ls[weakly_inds: weakly_inds + self.weakly_in_batch] + \
    #                     full_ls[full_inds: full_inds + self.full_in_batch]
    #             random.shuffle(batch)
    #             batch_ls.append(batch)
    #             weakly_inds += self.weakly_in_batch
    #             full_inds += self.full_in_batch
    #         else:
    #             batch = weakly_ls[weakly_inds: weakly_inds + self.weakly_in_batch] + \
    #                     full_ls[full_inds:] + full_ls[0: full_inds + self.full_in_batch - self.num_full]
    #             random.shuffle(batch)
    #             batch_ls.append(batch)
    #             weakly_inds += self.weakly_in_batch
    #             full_inds = full_inds + self.full_in_batch - self.num_full
    #
    #     if weakly_inds < self.num_weakly and (full_inds+self.batch_size-self.num_weakly+weakly_inds) <= self.num_full:
    #         batch = weakly_ls[weakly_inds:] + full_ls[full_inds:full_inds+self.batch_size-self.num_weakly+weakly_inds]
    #         random.shuffle(batch)
    #         batch_ls.append(batch)
    #     if weakly_inds < self.num_weakly and (full_inds+self.batch_size-self.num_weakly+weakly_inds) > self.num_full:
    #         batch = weakly_ls[weakly_inds:] + full_ls[full_inds:] + \
    #                 full_ls[0: full_inds+self.batch_size-self.num_weakly+weakly_inds - self.num_full]
    #         random.shuffle(batch)
    #         batch_ls.append(batch)
    #
    #     indices = [item for batch in batch_ls for item in batch]
    #     return iter(indices)

    # full supervised datas as major datas
    def __iter__(self):
        weakly_ls = list(range(self.num_weakly))
        full_ls = list(range(self.num_weakly, self.num_weakly + self.num_full))
        random.shuffle(weakly_ls)
        random.shuffle(full_ls)

        weakly_inds = 0
        full_inds = 0
        batch_ls = []
        while (full_inds + self.full_in_batch) <= self.num_full:
            if (weakly_inds + self.weakly_in_batch) <= self.num_weakly:
                batch = full_ls[full_inds: full_inds + self.full_in_batch] + \
                        weakly_ls[weakly_inds: weakly_inds + self.weakly_in_batch]
                random.shuffle(batch)
                batch_ls.append(batch)
                weakly_inds += self.weakly_in_batch
                full_inds += self.full_in_batch
            else:
                batch = full_ls[full_inds: full_inds + self.full_in_batch] + \
                        weakly_ls[weakly_inds:] + weakly_ls[0: weakly_inds + self.weakly_in_batch - self.num_weakly]
                random.shuffle(batch)
                batch_ls.append(batch)
                weakly_inds += self.weakly_in_batch
                full_inds = full_inds + self.full_in_batch - self.num_full

        if full_inds < self.num_full and (
                weakly_inds + self.batch_size - self.num_full + full_inds) <= self.num_weakly:
            batch = full_ls[full_inds:] + weakly_ls[
                                          weakly_inds:weakly_inds + self.batch_size - self.num_full + full_inds]
            random.shuffle(batch)
            batch_ls.append(batch)
        if full_inds < self.num_full and (
                full_inds + self.batch_size - self.num_full + weakly_inds) > self.num_weakly:
            batch = full_ls[full_inds:] + weakly_ls[weakly_inds:] + \
                    weakly_ls[0: weakly_inds + self.batch_size - self.num_full + weakly_full - self.num_weakly]
            random.shuffle(batch)
            batch_ls.append(batch)

        indices = [item for batch in batch_ls for item in batch]

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        return iter(indices)

    def __len__(self):
        return 1


class DistributedSeparateRandomSampler(Sampler):
    def __init__(self, num_weakly, num_full, batch_size, ratio_weakly, num_replicas=None, rank=None):
        # super(CombinedRandomSampler, self).__init__()
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil((num_weakly + num_full) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        assert batch_size >= 2, 'batch_size must be larger than 1, get {}'.format(batch_size)
        assert ratio_weakly < 1, 'ratio_weakly must be smaller than 1, get {}'.format(ratio_weakly)
        self.num_weakly = num_weakly
        self.num_full = num_full
        self.batch_size = batch_size

    def __iter__(self):
        weakly_ls = list(range(self.num_weakly))
        if self.overlap_anno_id:
            # random drop overlap instances
            for idx in self.overlap_anno_id:
                if random.random()<=0.75:
                    try:
                        weakly_ls.remove(idx)
                    except Exception:
                        pass
            print('sample {} weakly-supervised instances in this epoch.'.format(len(weakly_ls)))
        full_ls = list(range(self.num_weakly, self.num_weakly + self.num_full))
        random.shuffle(weakly_ls)
        random.shuffle(full_ls)

        num_batch = math.ceil((self.num_weakly + self.num_full) / self.batch_size)
        indices = []
        for i in range(num_batch):
            if i % 2 == 0:
                indices += weakly_ls[i / 2 * self.batch_size: i / 2 * self.batch_size + self.batch_size]
            else:
                indices += full_ls[(i - 1) / 2 * self.batch_size: (i - 1) / 2 * self.batch_size + self.batch_size]

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:len(indices):self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class CombinedSeparateSampler(Sampler):
    def __init__(self, num_weakly, num_full, batch_size):
        # super(CombinedRandomSampler, self).__init__()
        assert batch_size >= 2, 'batch_size must be larger than 1, get {}'.format(batch_size)
        self.num_weakly = num_weakly
        self.num_full = num_full
        self.batch_size = batch_size

    def __iter__(self):
        weakly_ls = list(range(self.num_weakly))
        full_ls = list(range(self.num_weakly, self.num_weakly + self.num_full))
        random.shuffle(weakly_ls)
        random.shuffle(full_ls)

        num_batch = math.ceil((self.num_weakly + self.num_full)/self.batch_size)
        indices = []
        for i in range(num_batch):
            if i%2==0:
                indices += weakly_ls[i/2*self.batch_size: i/2*self.batch_size+self.batch_size]
            else:
                indices += full_ls[(i-1)/2*self.batch_size: (i-1)/2*self.batch_size+self.batch_size]

        return iter(indices)

    def __len__(self):
        return 1


if __name__ == '__main__':
    datasource = range(350)
    a = CombinedRandomSampler(200, 150, 8, 0.75)
    for i, item in enumerate(a.__iter__()):
        print(i, item)


