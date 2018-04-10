#-*- coding:utf-8 -*-

import torch
import dataset
import numpy as np

import keys
characters = keys.alphabet[:]
from model import get_model

nclass = len(characters) + 1

import keras.backend as K

train_dataset = '../data/lmdb/train'
val_dataset = '../data/lmdb/val'
batchSize = 256
workers = 4
imgH = 32
imgW = 280  # 256
keep_ratio = False
random_sample = False


def one_hot(text, length=10, characters=characters):
     label = np.zeros(length)
     for i, char in enumerate(text.decode('utf-8')):
        index = characters.find(char)
        if index == -1:
            index = characters.find(u' ')
        label[i] = index
     return label

n_len = 10
def gen(loader, flag='train'):
    while True:
        i = 0 
        n = len(loader)
        for X, Y in loader:
            X = X.numpy()
            X = X.reshape((-1, imgH, imgW, 1))
            if flag=='test':
                Y = Y.numpy()
            
            Y = np.array(Y)
            Length = int(imgW / 4)-1
            batchs = X.shape[0]
            # Y = Y.numpy()
            if i > n-1:
                i = 0
                break
                
            yield [X, Y, np.ones(batchs) * int(Length), np.ones(batchs) * n_len], np.ones(batchs)

if random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, batchSize)
else:
    sampler = None

train_dataset = dataset.lmdbDataset(root=train_dataset, target_transform=one_hot)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(workers),
    collate_fn=dataset.alignCollate(imgH=imgH, imgW=imgW, keep_ratio=keep_ratio))

test_dataset = dataset.lmdbDataset(
    root=val_dataset, transform=dataset.resizeNormalize((imgW, imgH)),target_transform=one_hot)

test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, batch_size=batchSize, num_workers=int(workers))


if __name__=='__main__':
    import os
    if not os.path.exists('./models'):
        os.mkdir('./models')
    
    model, basemodel = get_model(height=imgH, nclass=nclass)
    
    '''
    modelPath = '../pretrain-models/keras.hdf5'
    if os.path.exists(modelPath):
        basemodel.load_weights(modelPath)
    '''

    ## 注意此处保存的是model的权重
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
    checkpointer = ModelCheckpoint(filepath="./models/model-{epoch:02d}-{val_loss:.2f}.h5", monitor='val_loss', verbose=0, save_weights_only=True, save_best_only=True)
    rlu = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

    model.fit_generator(gen(train_loader, flag = 'train'), 
                    steps_per_epoch = 3461807 / batchSize, 
                    epochs = 3,
                    validation_data = gen(test_loader, flag = 'test'),
                    callbacks = [checkpointer,rlu],
                    validation_steps = 182200 / batchSize)
