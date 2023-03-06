import os
import errno
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from configs import faults_idx
from utils import get_class, filter_key
from scipy.io import loadmat

class CWRUDataset(Dataset):
    def __init__(self, exps, rpms, length):
        super().__init__()
        self.length = length
        self.labels = []

        for exp in exps:
            if exp not in ('12DriveEndFault', '12FanEndFault', '48DriveEndFault'):
                print("Wrong experiment name: {}".format(exp))
                return
            
        for rpm in rpms:
            if rpm not in ('1797', '1772', '1750', '1730'):
                print("Wrong RPM value: {}".format(rpm))
                return

        rdir = os.path.join('Datasets/CWRU')
 
        fmeta = os.path.join(os.path.dirname('__file__'), 'metadata.txt')
        all_lines = open(fmeta).readlines()
        lines = []
        for line in all_lines:
            l = line.split()
            if (l[0] in exps or l[0] == 'NormalBaseline') and l[1] in rpms:
                if 'Normal' in l[2] or '0.007' in l[2] or '0.014' in l[2] or '0.021' in l[2]:
                    if faults_idx.get(l[2],-1)!=-1:
                        lines.append(l)

        lines = sorted(lines, key=lambda line: get_class(line[0],line[2])) 
        self.X, self.y = self._load_and_slice_data(rdir, lines)
        self.all_labels = tuple(((line[0]+line[2]),get_class(line[0],line[2])) for line in lines)
        self.classes = sorted(list(set(self.all_labels)), key=lambda label: label[1]) 
        self.nclasses = len(self.classes)  # number of classes

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        return self.X[index], self.y[index]
    

    def _mkdir(self, path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                print("can't create directory '{}''".format(path))
                exit(1)

    def _load_and_slice_data(self, rdir, infos):
            X = np.zeros((0, self.length, 2), dtype=np.float32)
            y = []

            print("loading dataset..")

            for idx, info in enumerate(infos):
    
                # directory of this file
                fdir = os.path.join(rdir, info[0], info[1])
                self._mkdir(fdir)
                fpath = os.path.join(fdir, info[2] + '.mat')
    
                mat_dict = loadmat(fpath)
                key1, key2 = filter_key(mat_dict.keys())
                time_series = np.hstack((mat_dict[key1], mat_dict[key2]))
                
                clips = np.zeros((0, 2))

                for cut in range(0, time_series.shape[0] - self.length + 1, self.length):
                    clips = np.vstack((clips, time_series[cut:cut+self.length]))
                clips = clips.reshape(-1, self.length, 2)
                X = np.vstack((X, clips))

                y += [get_class(info[0],info[2])] * clips.shape[0]
                
            X.reshape(-1, self.length, 2)

            print(f"datasets from {idx} file(s) loaded.")

            return X, y

    # for same dataset style with the original tensorflow code
    def _load_and_slice_data_2(self, rdir, infos):
            X = np.zeros((0, self.length, 2))
            y = []

            train_cuts = list(range(0,60000,80))[:660]
            test_cuts = list(range(60000,120000,self.length))[:25]

            print("loading dataset..")

            for idx, info in enumerate(infos):
    
                # directory of this file
                fdir = os.path.join(rdir, info[0], info[1])
                self._mkdir(fdir)
                fpath = os.path.join(fdir, info[2] + '.mat')
    
                mat_dict = loadmat(fpath)
                key1, key2 = filter_key(mat_dict.keys())
                time_series = np.hstack((mat_dict[key1], mat_dict[key2]))
                
                clips = np.zeros((0, 2))

                for cut in train_cuts:
                    clips = np.vstack((clips, time_series[cut:cut+self.length]))
                clips = clips.reshape(-1, self.length, 2)
                X = np.vstack((X, clips))

                clips = np.zeros((0, 2))

                for cut in test_cuts:
                    clips = np.vstack((clips, time_series[cut:cut+self.length]))
                clips = clips.reshape(-1, self.length, 2)
                X = np.vstack((X, clips))

                y += [get_class(info[0],info[2])] * clips.shape[0] * 685
                
            X.reshape(-1, self.length, 2)

            print(f"datasets from {idx} file(s) loaded.")

            return X, y

# make the same number of each labels in a batch.
def custom_collate_fn(batch): 
    num_zeros = 0
    num_ones = 0

    # Separate the data and labels in the batch
    data, labels = zip(*batch)
    batch_size = len(data)

    pairs = torch.empty((batch_size, 2, 2048, 2), dtype=torch.float32)
    targets = torch.empty((batch_size,), dtype=torch.float32)
    
    # Create pairs and targets
    for i in range(batch_size):
        data1, label1 = data[i], labels[i]
        
        # Select a data point with the same label (target 0)
        same_label_idxs = [j for j in range(batch_size) if labels[j] == label1 and j != i]
        if num_zeros < batch_size/2 and len(same_label_idxs) > 0:
            idx = random.choice(same_label_idxs)
            data2, label2 = data[idx], labels[idx]
            target = 0
            num_zeros += 1
        else:
            # Select a data point with a different label (target 1)
            diff_label_idxs = [j for j in range(batch_size) if labels[j] != label1]
            idx = random.choice(diff_label_idxs)
            data2, label2 = data[idx], labels[idx]
            target = 1
            num_ones += 1
        
        pairs[i][0] = torch.tensor(data1, dtype=torch.float32)
        pairs[i][1] = torch.tensor(data2, dtype=torch.float32)
        targets[i] = target

    x1, x2 = torch.split(pairs, split_size_or_sections=1, dim=3)
    x1 = torch.squeeze(x1, dim=3)
    x2 = torch.squeeze(x2, dim=3)
    targets = targets.clone().detach()

    return (x1, x2), targets



# when making a batch, randomly select two targets.
def custom_collate_fn_2(batch):
    
    # Unpack the batch into separate data points and labels
    data, labels = zip(*batch)
    batch_size = len(data)

    pairs = torch.empty((batch_size, 2, 2048, 2), dtype=torch.float32)
    targets = torch.empty((batch_size,), dtype=torch.float32)
    
    # Select two data points from each batch and create pairs
    for i in range(batch_size):
        data1, label1 = data[i], labels[i]
        random_number = random.randint(0, batch_size - 1)
        data2, label2 = data[random_number], labels[random_number]
        
        # Check if the two data points belong to the same class
        if label1 == label2:
            target = 0
        else:
            target = 1
        
        pairs[i][0] = torch.tensor(data1, dtype=torch.float32)
        pairs[i][1] = torch.tensor(data2, dtype=torch.float32)
        targets[i] = target

    x1, x2 = torch.split(pairs, split_size_or_sections=1, dim=3)
    x1 = torch.squeeze(x1, dim=3)
    x2 = torch.squeeze(x2, dim=3)
    targets = targets.clone().detach()
    
    return (x1, x2), targets