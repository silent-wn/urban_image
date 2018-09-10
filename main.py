import torch
import torch.nn as nn
import torchvision
from config import opt
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
import utils
import os
if not os.path.exists(opt.save):
        os.makedirs(opt.save)

from dumblog import dlog
trainLogger = dlog('train', logPath=os.path.join(opt.save, 'logs'))
valLogger = dlog('val', logPath=os.path.join(opt.save, 'logs'))




device = torch.device("cuda:{:d}".format(opt.gpu) if torch.cuda.is_available() else "cpu")

torch.manual_seed(opt.seed)



class Trainer():
    def __init__(self, opt):
        self.opt = opt
        # resetting model
        self.net = models.resnet18(pretrained=True)
        in_dim = self.net.fc.in_features
        self.net.fc = nn.Linear(in_dim, opt.classNum)
        self.net.to(device)
        # optimizer
        self.optim = torch.optim.Adam(self.net.parameters(), lr=opt.learningRate, betas=(opt.beta1, 0.999))

        # dataset
        self.dataset = datasets.ImageFolder(opt.dataRoot, 
                                            transform=transforms.Compose(
                                                [
                                                    transforms.Resize((224, 224)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                    ])
                                            )
        self.data_num = len(self.dataset)
        self.train_num = int(self.data_num*opt.trainFrac)

        # split dataset
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [self.train_num, self.data_num-self.train_num])
        self.trainDataloader = DataLoader(self.train_dataset, opt.batchSize, shuffle=True, num_workers=4)
        self.valDataloader = DataLoader(self.val_dataset, opt.batchSize, shuffle=False, num_workers=4)

        # loss function
        self.criterion = nn.CrossEntropyLoss()


    def train(self):
        self.net.train()
        total_loss = 0.0
        for i, (img, target) in enumerate(self.trainDataloader):
            img = img.to(device)
            target = target.to(device)
            self.optim.zero_grad()
            output = self.net(img)
            loss = self.criterion(output, target)
            loss.backward()
            self.optim.step()
            total_loss += loss.item()

        return total_loss/(i+i)

    def val(self):
        self.net.eval()
        acc = 0.0
        for i, (img, target) in enumerate(self.valDataloader):
            img = img.to(device)
            target = target.to(device)
            outputs = self.net(img)
            _, pred = torch.max(outputs, 1)
            acc += torch.sum(pred == target).item()
        return acc / len(self.valDataloader)



    def save_chkpt(self, is_best):
        utils.save_checkpoint(
            {
                'epoch': self.epoch_now,
                'net': self.net.state_dict(),
                'optim': self.optim.state_dict(),
                'best_acc': self.best_acc,
            }, is_best, 'checkpoint', self.opt.save
        )

    def load_chkpt(self, is_best):
        if is_best:
            filename = 'model_best.pth.tar'
        else:
            filename = 'checkpoint.pth.tar'
        model_path = os.path.join(opt.save, filename)
        try:
            checkpoint = torch.load(model_path)
            self.epoch_now = checkpoint['epoch']
            self.best_acc = checkpoint['best_acc']
            self.net.load_state_dict(checkpoint['net'])

            self.optim.load_state_dict(checkpoint['optim'])

            print('Success loading {}th epoch checkpoint!'.format(self.epoch_now))
        except:
            print('Failed to load checkpoint!')

    def run(self, resume=True, is_best=False):
        self.best_acc = 0
        self.epoch_now = 0

        if resume:
            self.load_chkpt(is_best=is_best)

        for i in range(self.epoch_now + 1 , opt.nEpochs):
            loss = self.train()
            trainLogger.info('Epoch {:d} Train Loss: {:.4f}'.format(i, loss))
            acc = self.val()
            valLogger.info('\t Val Loss: {:.4f}'.format(acc))
            self.epoch_now += 1

            if acc > self.best_acc:
                self.best_acc = acc
                print('Saving best model so far (val acc = {:.4f}) '.format(acc) + opt.save + '/model_best.pth.tar')
                self.save_chkpt(is_best=True)
            else:
                self.save_chkpt(is_best=False)



if __name__ == "__main__":
    trainer = Trainer(opt)
    trainer.run()