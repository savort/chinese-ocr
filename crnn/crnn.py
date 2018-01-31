#-*- coding:utf-8 -*-
import sys
sys.path.insert(1, "./crnn")
import torch
import torch.utils.data
from torch.autograd import Variable 
import numpy as np
import util
import dataset
import models.crnn as crnn
import keys
from math import *
import cv2
GPU = False


def crnnSource():
    alphabet = keys.alphabet
    converter = util.strLabelConverter(alphabet)
    if torch.cuda.is_available() and GPU:
       model = crnn.CRNN(32, 1, len(alphabet)+1, 256, 1).cuda()
    else:
        model = crnn.CRNN(32, 1, len(alphabet)+1, 256, 1).cpu()
    path = './crnn/samples/model_acc97.pth'
    model.eval()
    model.load_state_dict(torch.load(path))
    return model,converter

##加载模型
model,converter = crnnSource()

def crnnOcr(image):
       """
       crnn模型，ocr识别
       @@model,
       @@converter,
       @@im
       @@text_recs:text box

       """
       scale = image.size[1]*1.0 / 32
       w = image.size[0] / scale
       w = int(w)
       #print "im size:{},{}".format(image.size,w)
       transformer = dataset.resizeNormalize((w, 32))
       if torch.cuda.is_available() and GPU:
           image = transformer(image).cuda()
       else:
           image = transformer(image).cpu()
            
       image = image.view(1, *image.size())
       image = Variable(image)
       model.eval()
       preds = model(image)
       _, preds = preds.max(2)
       preds = preds.transpose(1, 0).contiguous().view(-1)
       preds_size = Variable(torch.IntTensor([preds.size(0)]))
       sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
       if len(sim_pred)>0:
          if sim_pred[0]==u'-':
             sim_pred=sim_pred[1:]

       return sim_pred

