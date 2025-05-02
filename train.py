import argparse
import logging
import os
import random
import sys
import torch
import glob
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils.dice_score import dice_coeff, multiclass_dice_coeff

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.data_loading import EyeglassDataset
from utils.dice_score import dice_loss

import os
import wandb
import torch
import logging
import numpy as np


class CheckpointSaver:
    def __init__(self, dirpath, wandb_enabled, decreasing=True, top_n=5):
        """
        dirpath: Directory path where to store all model weights
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models_dir to track based on validation metric value
        """
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        self.dirpath = dirpath
        self.wandb_enabled = wandb_enabled
        self.top_n = top_n
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf

    def __call__(self, model, epoch, metric_val):
        model_path = os.path.join(self.dirpath, model.__class__.__name__ + f'_epoch{epoch}.pt')
        save = metric_val < self.best_metric_val if self.decreasing else metric_val > self.best_metric_val
        if save:
            logging.info(
                f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}, & logging model weights to W&B.")
            self.best_metric_val = metric_val
            torch.save(model.state_dict(), model_path)
            if self.wandb_enabled:
                self.log_artifact(f'model-ckpt-epoch-{epoch}.pt', model_path, metric_val)
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)
        if len(self.top_model_paths) > self.top_n:
            self.cleanup()

    def log_artifact(self, filename, model_path, metric_val):
        artifact = wandb.Artifact(filename, type='model', metadata={'Validation score': metric_val})
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)

    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        logging.info(f"Removing extra models_dir.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')

import wandb
#wandb.login()
wandb.login(key="1da3729d6dfaff15faa7afc015bfa5857dfc0b59")  # Correct method


eyedataset_train = None
lr_type = 'none'
def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError, IndexError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)
    #
    # # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    #
    # # 3. Create data loaders
    # loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    # train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    # val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    global eyedataset_train

    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)

    base_dir = '/data/image_databases/detection_benchmark/gt_boxing'

    # Initialize lists for train and test directories
    train_dirs = []
    test_dirs = []

    # Traverse the directories in the base path
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)

        # Ensure it's a directory
        if os.path.isdir(folder_path):
            # Check if it's a train folder (contains '_train' but not 'metahuman')
            if '_train' in folder or 'metahuman'  in folder:
                train_dirs.append(folder_path)
            # Check if it's a test folder (contains '_test' but not 'metahuman')
            elif '_test' in folder:
                test_dirs.append(folder_path)

    #train_img_dir = '/data/image_databases/detection_benchmark/gt_boxing/near_annotated_2024_train'
    #test_img_dir = '/data/image_databases/detection_benchmark/gt_boxing/near_annotated_2024_test'

    train_img_dirs = []
    for train_dir in train_dirs:
        train_img_dirs.extend(glob.glob(f"{train_dir}/*/"))

    if not train_img_dirs:  # If no subfolders, use the main directory
        train_img_dirs = train_dirs

    #print(f"train_img_dirs: {train_img_dirs}, type: {type(train_img_dirs)}")

    eyedataset_train = EyeglassDataset(
        image_dir=train_img_dirs,
        augment=True
    )

    #print(f"test_dirs: {test_dirs}, type: {type(test_dirs)}")

    eyedataset_val = EyeglassDataset(
    image_dir=test_dirs,
    augment=False)

    if not train_dirs:
        raise ValueError("No valid train directories found.")
    if not test_dirs:
        raise ValueError("No valid test directories found.")


    # Create train_loader and val_loader

    # Arguments for DataLoader
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)

    train_loader = DataLoader(eyedataset_train, shuffle=True, **loader_args)
    val_loader = DataLoader(eyedataset_val, shuffle=False, **loader_args)

    n_train=len(eyedataset_train)
    n_val=len(eyedataset_val)

    # (Initialize logging)
    wandb.login(key='1da3729d6dfaff15faa7afc015bfa5857dfc0b59')
    experiment = wandb.init(project='U-Net-eyeglasses', name=f"run_lr{learning_rate}_epochs{epochs}")
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp,
             scheduler=lr_type)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = None
    if lr_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    elif lr_type == 'stepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1, last_epoch=-1)
    elif lr_type == 'none':
        scheduler = None
    else:
        raise ValueError("Unknown scheduler")




    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    val_dice_scores = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        dice_score = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            num_images = 0
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                num_images += images.shape[0]
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )



                # Calculate Dice score
                if model.n_classes == 1:
                    masks_pred = (F.sigmoid(masks_pred) > 0.5).float()
                    dice_score += dice_coeff(masks_pred, true_masks, reduce_batch_first=False)
                else:
                    masks_pred = F.one_hot(masks_pred.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()
                    #masks_pred = masks_pred.squeeze()
                    #print(f"Before squeeze: masks_pred shape: {masks_pred.shape}, true_masks shape: {true_masks.shape}")
                    masks_pred = masks_pred.squeeze(1)
                    masks_pred = masks_pred.argmax(dim=1, keepdim=False)
                    #print(f"After squeeze: masks_pred shape: {masks_pred.shape}, true_masks shape: {true_masks.shape}")

                    # print(masks_pred[:, 1:].shape,  true_masks[:, 1:].shape)

                    #####
                    dice_score += multiclass_dice_coeff(masks_pred[:, 1:], true_masks[:, 1:],
                                                        reduce_batch_first=False)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                dice_score = dice_score / max(num_images, 1)
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'train Dice': dice_score, #dice_score, # sa modific dupa evaluate.py
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})


                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score, validation_loss = evaluate(model, val_loader, device, amp)
                        if scheduler:
                            scheduler.step()

                        logging.info('Validation Dice score: {}'.format(val_score))

                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'validation loss': validation_loss,

                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(masks_pred[0].unsqueeze(0).float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })



        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = eyedataset_train.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5, #modific in rularea 2
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()


    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )



