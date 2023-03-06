import os
import errno
import random
import numpy as np
from scipy.io import loadmat
import urllib.request as urllib
from sklearn.utils import shuffle

from configs import faults_idx
from utils import filter_key, get_class

class load_dataset:
    def __init__(self, exps, rpms, length):
        for exp in exps:
            if exp not in ('12DriveEndFault', '12FanEndFault', '48DriveEndFault'):
                print("wrong experiment name: {}".format(exp))
                return
        for rpm in rpms:    
            if rpm not in ('1797', '1772', '1750', '1730'):
                print("wrong rpm value: {}".format(rpm))
                return
        # root directory of all data
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
 
        self.length = length  # sequence length
        lines = sorted(lines, key=lambda line: get_class(line[0],line[2])) 
        self._load_and_slice_data(rdir, lines)
        # shuffle training and test arrays
        self._shuffle()
        self.all_labels = tuple(((line[0]+line[2]),get_class(line[0],line[2])) for line in lines)
        self.classes = sorted(list(set(self.all_labels)), key=lambda label: label[1]) 
        self.nclasses = len(self.classes)  # number of classes
 
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
        self.X_train = np.zeros((0, self.length, 2))
        self.X_test = np.zeros((0, self.length, 2))
        self.y_train = []
        self.y_test = []
        train_cuts = list(range(0,60000,80))[:660]
        test_cuts = list(range(60000,120000,self.length))[:25]

        print("loading datasets...")
        for idx, info in enumerate(infos):
 
            # directory of this file
            fdir = os.path.join(rdir, info[0], info[1])
            self._mkdir(fdir)
            fpath = os.path.join(fdir, info[2] + '.mat')

            if not os.path.exists(fpath):
                self._download(fpath, info[3].rstrip('\n'))
 
            mat_dict = loadmat(fpath)
            key1,key2 = filter_key(mat_dict.keys())
            time_series = np.hstack((mat_dict[key1], mat_dict[key2]))
            idx_last = -(time_series.shape[0] % self.length)
            
            clips = np.zeros((0, 2))

            for cut in shuffle(train_cuts, random_state=0): # set seed as 0
                clips = np.vstack((clips, time_series[cut:cut+self.length]))
            clips = clips.reshape(-1, self.length,2)
            self.X_train = np.vstack((self.X_train, clips))
            
            clips = np.zeros((0, 2))

            for cut in shuffle(test_cuts, random_state=0): # set seed as 0
                clips = np.vstack((clips, time_series[cut:cut+self.length]))
            clips = clips.reshape(-1, self.length,2)
            self.X_test = np.vstack((self.X_test, clips))
            
            self.y_train += [get_class(info[0],info[2])] * 660
            self.y_test += [get_class(info[0],info[2])] * 25
            
        self.X_train.reshape(-1, self.length,2)
        self.X_test.reshape(-1, self.length,2)

        print(len(infos), "data files loaded and sliced into train and test sets.")
 
    def _shuffle(self):
        # shuffle training samples
        index = list(range(self.X_train.shape[0]))
        random.Random(0).shuffle(index)
        self.X_train = self.X_train[index]
        self.y_train = np.array(tuple(self.y_train[i] for i in index))
 
        # shuffle test samples
        index = list(range(self.X_test.shape[0]))
        random.Random(0).shuffle(index)
        self.X_test = self.X_test[index]
        self.y_test = np.array(tuple(self.y_test[i] for i in index))