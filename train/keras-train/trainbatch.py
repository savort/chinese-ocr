#-*- coding:utf-8 -*-
import os
from train import *
model, basemodel = get_model(height=imgH, nclass=nclass)

'''
modelPath = '../pretrain-models/keras.hdf5'
if os.path.exists(modelPath):
    basemodel.load_weights(modelPath)
'''

if not os.path.exists('./models'):
    os.mkdir('./models')

batchSize = 128
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(workers),
    collate_fn=dataset.alignCollate(imgH=imgH, imgW=imgW, keep_ratio=keep_ratio))

testSize = 128
# print test_dataset[0]
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=testSize,
    shuffle=True, num_workers=int(workers))

step = 0
n_epoch = 3
interval = 50
best_loss = 1000
current_loss = 'NaN'
for step in range(n_epoch):
    for X, Y in train_loader:
        X = X.numpy()
        X = X.reshape((-1, imgH, imgW, 1))
        Y = np.array(Y)
                
        Length = int(imgW / 4) - 2
        batch = X.shape[0]
        X, Y = [X, Y, np.ones(batch) * Length, np.ones(batch) * n_len], np.ones(batch)    
        model.train_on_batch(X, Y)
        if step % interval == 0 :
            X, Y = next(iter(test_loader))
            X = X.numpy()
            X = X.reshape((-1, imgH, imgW, 1))
            Y = Y.numpy()
            Y = np.array(Y)
            batch = X.shape[0]
            X, Y = [X, Y, np.ones(batch) * Length, np.ones(batch) * n_len], np.ones(batch) 

            current_loss = model.evaluate(X, Y)
            print("global_steps: {}, best_loss: {}, current_loss: {}, current_acc: {}".format(step, best_loss, current_loss[0], current_loss[1]))
            if current_loss[0] < best_loss:
                best_loss = current_loss[0]
                path = './models/model_{}.h5'.format(best_loss)
                print("get a best loss, model saving to {}...".format(path))
                basemodel.save(path)

        step += 1

