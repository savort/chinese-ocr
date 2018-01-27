#-*- coding:utf-8 -*-
import os
import sys
import time
import model
import numpy as np
from PIL import Image
from glob import glob
paths = glob('./test/*.*')


if __name__ =='__main__':
    im = Image.open(paths[0])
    img = np.array(im.convert('RGB'))
    # img = img[:, :, ::-1]  # Convert RGB to BGR
    t = time.time()
    result, img, angle = model.model(img, model='torch', detectAngle=True)
    Image.fromarray(img).save('result.jpg')
    print("Mission complete, it takes {}s".format(time.time() - t))
    print("---------------------------------------")
    print("The literal orientation of the image is {} degrees".format(angle))
    print("Recognition Result:\n")
    for key in result:
        print result[key][1]

