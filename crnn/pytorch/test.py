#-*- coding:utf-8 -*-
import os
import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import keys
import crnn
import dataset
from utils import strLabelConverter


alphabet = keys.alphabet
raw_input('\ninput: ')
converter = strLabelConverter(alphabet)
model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1).cuda()
path = './models/model_acc97.pth'
model.load_state_dict(torch.load(path))
print(model)

while 1:
    im_name = raw_input("\nplease input file name: ")
    im_path = "./img/" + im_name
    image = Image.open(im_path).convert('L')
    scale = image.size[1] * 1.0 / 32
    w = int(image.size[0] / scale)

    transformer = dataset.resizeNormalize((w, 32))
    image = transformer(image).cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    model.eval()
    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.squeeze(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))

