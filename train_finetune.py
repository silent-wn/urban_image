# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler 
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
from global_config import *
from dataloader.dataloader import DataLoader
from model.ft_res18_model import fine_tune_model
from visdom import Visdom


def train_model(data_loader, model, criterion, optimizer, scheduler, num_epochs=50):
    since_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model = model
    best_acc = 0
    epoch = 0
    loss = 0
    acc = 0
    viz_loss = Visdom(env = 'loss')
    viz_acc = Visdom(env = 'acc')

    win_acc = viz_acc.line(
        X=np.array([epoch]),
        Y=np.array([acc]),
        opts=dict(title='acc')
    )

    win_loss = viz_loss.line(
        X=np.array([epoch]),
        Y=np.array([loss]),
        opts=dict(title='loss')
    )

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)
            running_loss = 0.0
            running_corrects = 0
            for batch_data in data_loader.load_data(data_set=phase):
                inputs, labels = batch_data
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print('inputs2=',inputs.size())
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, predict = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(predict == labels.data)

            epoch_loss = running_loss / data_loader.data_sizes[phase]
            epoch_acc = running_corrects.double() / data_loader.data_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)
        
        
        #viz可视化   

        # win = viz_loss.line(
        #     X=np.array([x]),
        #     Y=np.array([y]),
        #     opts=dict(title='loss')
        # )
        # x = epoch
        # y = epoch_loss
        # viz_loss.line(
        #     X=np.array([x]),
        #     Y=np.array([y]),
        #     win = win,
        #     update = 'append'
        # )
        
        # x, y = 0
        # x = epoch
        # y = epoch_acc
        viz_acc.line(
            X=np.array([epoch]),
            Y=np.array([epoch_acc]),
            win = win_acc,
            update = 'append'
        )

        # viz_loss.line(
        #     X=np.array([epoch]),
        #     Y=np.array([epoch_loss]),
        #     win = win_loss,
        #     update = 'append'
        # )


        print()
    time_elapsed = time.time() - since_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model

# def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
#     """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
#     lr = init_lr * (0.1**(epoch // lr_decay_epoch))

#     if epoch % lr_decay_epoch == 0:
#         print('LR is set to {}'.format(lr))

#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

#     return optimizer

def save_torch_model(model, name):
    torch.save(model.state_dict(), name)

def train():
    data_loader = DataLoader(data_dir='/media/DATASET/wangning/data/建筑材质', image_size=IMAGE_SIZE, batch_size=64)
    inputs, classes = next(iter(data_loader.load_data()))
    out = torchvision.utils.make_grid(inputs)
    data_loader.show_image(out, title = [data_loader.data_classes[c] for c in classes])
    
    model = fine_tune_model()
    criterion = nn.CrossEntropyLoss()
    ignored_params = list(map(id, model.fc.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                     model.parameters())
    optimizer_ft = optim.SGD([
            {'params': base_params},
            {'params': model.fc.parameters(), 'lr': 1e-2}
            ], lr=1e-3, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    try:
        model = train_model(data_loader, model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)
        save_torch_model(model, MODEL_SAVE_FILE)
    except KeyboardInterrupt:
        print('manually interrupt, try saving model for now...')
        save_torch_model(model, MODEL_SAVE_FILE)
        print('model saved.')

if __name__ == '__main__':  
    train()