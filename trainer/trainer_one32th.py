import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from trainer.eval import eval_net

from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import BasicDataset
from torch.utils.data import DataLoader
from models.pan_regnety120 import PAN

import numpy as np
import random

torch.set_deterministic(True)
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# ailab
train_images_path = "/data.local/all/hangd/dynamic_data/one32rd/imgs/"
train_mask_path = "/data.local/all/hangd/dynamic_data/one32rd/masks/"

test_image_path = '/data.local/all/hangd/src_code_3/Pytorch-UNet/data_test/imgs/'
test_mask_path = '/data.local/all/hangd/src_code_3/Pytorch-UNet/data_test/masks/'


def train_net(
        dir_checkpoint,
        n_classes,
        n_channels,
        device,
        epochs=1,
        save_cp=True,
        img_scale=1):
    best_test_iou_score = 0.

    net = PAN()
    net.to(device=device)
    if os.path.exists(args.checkpoint_path):
        net.load_state_dict(
            torch.load(args.checkpoint_path, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    train_dataset = BasicDataset(train_images_path, train_mask_path, img_scale)
    test_dataset = BasicDataset(imgs_dir=test_image_path, masks_dir=test_mask_path, train=False, scale=img_scale)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    writer = SummaryWriter(comment=f'_{net.__class__.__name__}_LR_{args.lr}_BS_{args.batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Checkpoints:     {save_cp}
        Device:          {device}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    if n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        n_train = len(train_dataset)
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == n_channels, \
                    f'Network has been defined with {n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if n_classes == 1 else torch.long

                true_masks = true_masks.to(device=device, dtype=mask_type)
                masks_pred = net(imgs)  # return BCHW = 8_1_256_256
                _tem = net(imgs)
                # print("IS DIFFERENT OR NOT: ", torch.sum(masks_pred - _tem))

                true_masks = true_masks[:, :1, :, :]
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update(imgs.shape[0])
                global_step += 1

        # Tính dice và iou score trên tập Test set, ghi vào tensorboard .
        test_score_dice, test_score_iou = eval_net(net, test_loader, n_classes, device)
        if test_score_iou > best_test_iou_score:
            best_test_iou_score = test_score_iou
            os.makedirs(dir_checkpoint, exist_ok=True)
            torch.save(net.state_dict(), dir_checkpoint + f'best_CP_epoch{epoch + 1}_one32th_.pth')

        logging.info('Test Dice Coeff: {}'.format(test_score_dice))
        print('Test Dice Coeff: {}'.format(test_score_dice))
        writer.add_scalar('Dice/test', test_score_dice, epoch)

        logging.info('Test IOU : {}'.format(test_score_iou))
        print('Test IOU : {}'.format(test_score_iou))
        writer.add_scalar('IOU/test', test_score_iou, epoch)

    print("best iou: ", best_test_iou_score)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--method', type=str, default='i',
                        help='Choose dropout method: i for MCdropout ; w for Dropconnect')
    parser.add_argument('--cuda', type=int, nargs='?', default=1,
                        help='index of cuda', dest='cuda_inx')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('-lr', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--checkpoint_path', type=str,
                        default="",
                        help='Load model from a .pth file')
    parser.add_argument('--scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('--validation', type=float, default=10.0,
                        help='Percent of the dataset that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    dir_ckp = "./out/"
    device = torch.device("cpu") if not torch.cuda.is_available() else "cuda:0"
    logging.info(f'Using device {device}')

    n_classes = 1
    n_channels = 3

    logging.info(f'Network:\n'
                 f'\t{n_channels} input channels\n'
                 f'\t{n_classes} output channels (classes)\n'
                 )

    try:
        train_net(dir_checkpoint=dir_ckp,
                  n_classes=n_classes,
                  n_channels=n_channels,
                  epochs=args.epochs,
                  device=device,
                  img_scale=args.scale, )
    except KeyboardInterrupt:
        logging.info('Saved interrupt')
        exit(0)
