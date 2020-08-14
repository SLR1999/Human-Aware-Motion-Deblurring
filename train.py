from human_aware import HumanAware
from data_loader import DocumentDeblurrDataset
import torch
from metrics import PSNR, SSIM
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, ConvTranspose2d, Sigmoid, MSELoss
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import gc
import copy
import time
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from collections import defaultdict


class Trainer:

    def __init__(self, model, optimizer, criterion, scheduler, num_epochs=25, batch_size=1):
        gc.collect()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.use_gpu = torch.cuda.is_available()
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if self.use_gpu:
            self.model.cuda()
        self.num_epochs = num_epochs
        self.load_dataset(batch_size)

    def loss_with_attention(self, output, target, attention_map):
        loss = torch.mean((torch.mul(output,attention_map) - torch.mul(target,attention_map))**2)
        return loss

    def load_dataset(self, batch_size):
        # print ("Need to write")
        train_set = DocumentDeblurrDataset("./data/train/blurred_images/",
                               "./data/train/clear_images/",
                               "./data/train/attention_maps/",
                               self.transform)
        val_set = DocumentDeblurrDataset("./data/val/blurred_images/",
                               "./data/val/clear_images/",
                               "./data/val/attention_maps/",
                               self.transform)
        self.image_datasets = {
            'train': train_set, 'val': val_set
        }

        self.dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
            'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
        }

    def calculate_loss(self, outputs, real_images, criterion, metrics):
        loss = self.criterion(outputs, real_images)
        # metrics['PSNR'] = PSNR(outputs, real_images)
        metrics['SSIM'] += SSIM(outputs, real_images).data.cpu().numpy() * real_images.size(0)
        metrics['Loss'] += loss.data.cpu().numpy() * real_images.size(0)
        return loss

    def calculate_fg_loss(self, outputs, real_images, attention_map, criterion, metrics):
        loss = self.loss_with_attention(outputs, real_images, attention_map)
        # metrics['PSNR'] = PSNR(outputs, real_images)
        metrics['SSIM'] += SSIM(outputs, real_images).data.cpu().numpy() * real_images.size(0)
        metrics['Loss'] += loss.data.cpu().numpy() * real_images.size(0)
        return loss

    def calculate_bg_loss(self, outputs, real_images, attention_map, criterion, metrics):
        loss = self.loss_with_attention(outputs, real_images, 1-attention_map)
        # metrics['PSNR'] = PSNR(outputs, real_images)
        metrics['SSIM'] += SSIM(outputs, real_images).data.cpu().numpy() * real_images.size(0)
        metrics['Loss'] += loss.data.cpu().numpy() * real_images.size(0)
        return loss

    def print_metrics(self, metrics, epoch_samples, phase):
        outputs = []
        for k in metrics.keys():
            outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

        print("{}: {}".format(phase, ", ".join(outputs)))

    def train(self):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 1e10

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            since = time.time()

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                metrics = defaultdict(float)
                epoch_samples = 0
                for blurred_images, real_images, attention_maps in self.dataloaders[phase]:
                    if self.use_gpu:
                        blurred_images = blurred_images.cuda()
                        real_images = real_images.cuda()

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        p_decoder_output, fg_decoder_output, bg_decoder_output = self.model(blurred_images)
                        p_loss = self.calculate_loss(p_decoder_output, real_images, self.criterion, metrics)
                        fg_loss = self.calculate_fg_loss(fg_decoder_output, real_images, attention_maps, self.criterion, metrics)
                        bg_loss = self.calculate_bg_loss(bg_decoder_output, real_images, attention_maps, self.criterion, metrics)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            p_loss.backward()
                            fg_loss.backward()
                            bg_loss.backward()
                            self.optimizer.step()

                    # statistics
                    epoch_samples += blurred_images.size(0)

                if phase == 'train':
                    self.scheduler.step()

                self.print_metrics(metrics, epoch_samples, phase)
                epoch_loss = metrics['loss'] / epoch_samples

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    print("saving best model")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            time_elapsed = time.time() - since
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        print('Best val loss: {:4f}'.format(best_loss))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        gc.collect()

if __name__ == "__main__":

    humanAware = HumanAware()
    criterion = MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, humanAware.parameters()), lr=1e-3)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    trainer = Trainer(humanAware,optimizer, criterion, exp_lr_scheduler)
    trainer.train()