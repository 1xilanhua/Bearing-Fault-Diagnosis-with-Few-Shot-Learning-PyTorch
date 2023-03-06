import torch
from torchsummary import summary

from dataset import CWRUDataset
from models import SiameseNet
from configs import window_size

exp_list = ['12DriveEndFault']
rpm_list = ['1772', '1750', '1730']

data = CWRUDataset(exp_list, rpm_list, window_size)

model = SiameseNet()
if torch.cuda.is_available():
    model = model.to('cuda')

summary(model, torch.zeros((1, 2, 2048)), torch.zeros((1, 2, 2048)))