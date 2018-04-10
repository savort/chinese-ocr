#-*- coding:utf-8 -*-
from keras.applications.vgg16 import preprocess_input,VGG16
from keras.layers import Dense
from keras.models import Model
import numpy as np
from PIL import Image
from keras.optimizers import SGD


def load_model():
    sgd = SGD(lr=0.00001, momentum=0.9)
    model = VGG16(weights=None, classes=4)
    # 加载模型权重
    model.load_weights('./angle/models/modelAngle.h5', by_name=True)
    # 编译模型，以较小的学习率进行训练
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 加载模型
model = load_model()

def predict(path=None, img=None):
    """
    图片文字方向预测
    """
    ROTATE = [0, 270, 180, 90]
    if path is not None:
       im = Image.open(path).convert('RGB')
    elif img is not None:
       im = Image.fromarray(img).convert('RGB')
    w, h = im.size
    thesh = 0.05
    xmin, ymin, xmax, ymax = int(thesh * w), int(thesh * h), w - int(thesh * w), h - int(thesh * h)
    im = im.crop((xmin, ymin, xmax, ymax))  # 剪切图片边缘，清楚边缘噪声
    im = im.resize((224, 224))
    img = np.array(im)
    img = preprocess_input(img.astype(np.float32))
    pred = model.predict(np.array([img]))
    index = np.argmax(pred, axis=1)[0]
    return ROTATE[index]

