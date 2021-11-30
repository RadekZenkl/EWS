from __future__ import print_function, division

from methods import base
import segmentation_models_pytorch as smp
import torch.optim
import torch.nn as nn
import glob
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

import torch
import numpy as np
import torch.backends.cudnn

from helpers.helper_fcts import PadInput, RandomCrop, EncodedToTensor, RandomRotate,\
    RandomHorizontalFlip, RandomVerticalFlip, Normalize, iou, EWS_Dataset


class DLv3Plus(base.BenchmarkMethod):
    def __init__(self, pretrained):
        super().__init__()

        self.model = deeplabv3plus(pretrained=pretrained)

        self.epochs = 150
        self.batchsize = 16
        self.eval_batchsize = 4

        self.lr = 0.1
        self.momentum = 0.9
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(lr=self.lr, momentum=self.momentum, params=self.model.parameters())

        self.n_workers = 8
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.best_score = 0

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    def preprocess_image(self, image: np.array):
        pass

    def train(self, train_path='data/train', val_path='data/validation'):
        # Get all the paths to training images
        train_paths = glob.glob(train_path + '/*6.png')  # This rejects all *mask.png paths
        train_dataset = EWS_Dataset(train_paths, self.get_transforms(testval=False))
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.batchsize,
                                      shuffle=True,
                                      num_workers=self.n_workers,
                                      pin_memory=True)

        # Get all the paths to validation images
        val_paths = glob.glob(val_path + '/*6.png')  # This rejects all *mask.png paths
        val_dataset = EWS_Dataset(val_paths, self.get_transforms(testval=True))
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=int(self.eval_batchsize),
                                    shuffle=False,
                                    num_workers=self.n_workers,
                                    pin_memory=True)

        self.print_run_info()
        self.model.to(self.device)

        scaler = GradScaler()

        for epoch in tqdm(range(self.epochs), desc="Training \t", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            temp_score = 0

            preds_logged_train = torch.zeros((len(train_dataset), 224 * 224))
            y_logged_train = torch.zeros((len(train_dataset), 224 * 224))

            self.model.train()
            self.model.train_size()

            for i, (X, Y) in enumerate(train_dataloader):

                X = {key: X[key].to(self.device, non_blocking=True) for key in ['image', 'iso', 'fnumber', 'exposure']}
                Y = {key: Y[key].to(self.device, non_blocking=True) for key in ['mask']}

                # Runs the forward pass with autocasting for mixed precission.
                with autocast():
                    # forward + backward + optimize
                    outputs = self.model(X)
                    loss = self.criterion(outputs['mask'], Y['mask'].long().squeeze(1))

                for param in self.model.parameters():
                    param.grad = None

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                # log predictions and labels for metrics later
                n_samples = Y['mask'].shape[0]
                preds = torch.argmax(outputs['mask'], dim=1).reshape((n_samples, -1))
                preds_logged_train[i * self.batchsize: i * self.batchsize + n_samples] = preds.cpu()
                y_logged_train[i * self.batchsize: i * self.batchsize + n_samples] = Y['mask'].reshape(
                    (n_samples, -1)).cpu()

            del loss, outputs, preds

            # Track a given validation metric to keep the best performing model
            self.model.eval()
            self.model.eval_size()

            for i, (X, Y) in enumerate((val_dataloader)):
                X = {key: X[key].to(self.device, non_blocking=True) for key in ['image']}
                Y = {key: Y[key].to(self.device, non_blocking=True) for key in ['mask']}

                # Runs the forward pass with autocasting for mixed precision.
                with autocast():
                    outputs = self.model(X)

                # remove the image pads
                outputs['mask'] = outputs['mask'][:, :, :350, :350]

                # log predictions and labels for metrics later
                temp_iou = iou('mask', outputs, Y, self.device, 0).sum().cpu()
                temp_score += temp_iou[temp_iou.isfinite()].sum().numpy()

            temp_score /= len(val_dataset)

            # save best performing model
            if temp_score > self.best_score:
                self.best_score = temp_score
                path = 'best_model.pth'
                torch.save({
                    'model_state_dict': self.model.state_dict()
                }, path)

            del outputs

        train_metrics = self.calculate_metrics(preds_logged_train, y_logged_train)

        preds_logged_val = torch.zeros((len(val_dataset), 350 * 350))
        y_logged_val = torch.zeros((len(val_dataset), 350 * 350))

        self.load_model(model_name='best_model.pth')
        self.model.eval()
        self.model.eval_size()

        for i, (X, Y) in enumerate(tqdm(val_dataloader, desc="Validation ")):
            X = {key: X[key].to(self.device, non_blocking=True) for key in ['image']}
            Y = {key: Y[key].to(self.device, non_blocking=True) for key in ['mask']}

            # Runs the forward pass with autocasting for mixed precission.
            with autocast():
                # forward + backward + optimize
                outputs = self.model(X)

            # remove the image pads
            outputs['mask'] = outputs['mask'][:, :, :350, :350]

            # log predictions and labels for metrics later
            n_samples = Y['mask'].shape[0]
            preds = torch.argmax(outputs['mask'], dim=1).reshape((n_samples, -1))
            preds_logged_val[i * self.eval_batchsize: i * self.eval_batchsize + n_samples] = preds.cpu()
            y_logged_val[i * self.eval_batchsize: i * self.eval_batchsize + n_samples] = Y['mask'].reshape(
                (n_samples, -1)).cpu()

        val_metrics = self.calculate_metrics(preds_logged_val, y_logged_val)
        return train_metrics, val_metrics

    def test(self, test_path='data/test'):
        # Get all the paths to test images
        test_paths = sorted(glob.glob(test_path + '/*6.png'))  # This rejects all *mask.png paths
        test_dataset = EWS_Dataset(test_paths, self.get_transforms(testval=True))
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=int(self.eval_batchsize),
                                     shuffle=False,
                                     num_workers=self.n_workers,
                                     pin_memory=True)

        self.load_model(model_name='best_model.pth')

        preds_logged_test = torch.zeros((len(test_dataset), 350 * 350))
        y_logged_test = torch.zeros((len(test_dataset), 350 * 350))

        self.model.eval()
        self.model.eval_size()

        for i, (X, Y) in enumerate(tqdm(test_dataloader, desc="Testing ")):
            X = {key: X[key].to(self.device, non_blocking=True) for key in ['image']}
            Y = {key: Y[key].to(self.device, non_blocking=True) for key in ['mask']}

            # Runs the forward pass with autocasting for mixed precission.
            with autocast():
                # forward + backward + optimize
                outputs = self.model(X)

            # remove the image pads
            outputs['mask'] = outputs['mask'][:, :, :350, :350]

            # log predictions and labels for metrics later
            n_samples = Y['mask'].shape[0]
            preds = torch.argmax(outputs['mask'], dim=1).reshape((n_samples, -1))
            preds_logged_test[i * self.eval_batchsize: i * self.eval_batchsize + n_samples] = preds.cpu()
            y_logged_test[i * self.eval_batchsize: i * self.eval_batchsize + n_samples] = Y['mask'].reshape(
                (n_samples, -1)).cpu()

        test_metrics = self.calculate_metrics(preds_logged_test, y_logged_test)
        return test_metrics, preds_logged_test, y_logged_test

    def load_model(self, model_name='best_model.pth'):
        if self.device == 'cuda:0':
            checkpoint = torch.load(model_name, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
        else:
            checkpoint = torch.load(model_name, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.to(self.device)
        return

    def print_run_info(self):
        """
        Print information about training
        Returns:

        """
        print()
        print("-" * 80)
        print('training for ', self.epochs, ' epochs')
        print()
        print('Starting training of the model')
        print('Training is using: ' + str(self.device))
        print(torch.cuda.get_device_name(0))
        print("-" * 80)
        print()

    def get_transforms(self, testval: bool):

        transforms = []
        if testval:
            transforms.append(EncodedToTensor())
            transforms.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
            transforms.append(PadInput((352, 352)))

        else:
            transforms.append(EncodedToTensor())
            transforms.append(RandomRotate(20))
            transforms.append(RandomCrop((224, 224)))
            transforms.append(RandomHorizontalFlip())
            transforms.append(RandomVerticalFlip())
            transforms.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

        return Compose(transforms)


def deeplabv3plus(pretrained):
    if pretrained:
        network = smp.DeepLabV3Plus('resnet50', classes=2)
    else:
        network = smp.DeepLabV3Plus('resnet50', classes=2, encoder_weights=None)

    name = 'dlb50_inj50ft15'
    inputs = ['image']
    outputs = ['mask']

    model = DeepLabV3Plus(name, inputs, outputs, network)

    return model


class DeepLabV3Plus(nn.Module):
    def __init__(self, name, inputs, outputs, network):
        super().__init__()
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.network = network
        self.injections = torch.nn.ModuleDict()
        self.evaluation = False
        self.eval_injections = torch.nn.ModuleDict()

    def eval_size(self):
        self.evaluation = True

    def train_size(self):
        self.evaluation = False

    def forward(self, x):
        # keep inputs for injection, drop images
        x = x['image']

        masks = self.network(x)

        return {'mask': masks}
