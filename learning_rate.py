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
import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.data_loading import EyeglassDataset
from utils.dice_score import dice_loss


#evaluate


# Directories for images, masks, and checkpoints
dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')

# Initialize wandb
wandb.login()

# Supported image extensions
image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}

# Helper function to count images in a folder
def count_images(folder_path):
    """Count the number of valid images in a folder."""
    if not os.path.isdir(folder_path):
        return 0
    files = os.listdir(folder_path)
    return sum(1 for f in files if os.path.splitext(f)[-1].lower() in image_extensions)

# Main training function
def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # Set loader arguments
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)

    # Base directory containing dataset folders
    base_dir = '/data/image_databases/detection_benchmark/gt_boxing'

    # Initialize lists for train and test directories
    train_dirs = []
    test_dirs = []

    # Traverse directories in the base path
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)

        # Ensure it's a directory
        if os.path.isdir(folder_path):
            # Check for train folders (contains '_train')
            if '_train' in folder:
                train_dirs.append(folder_path)
            # Check for test folders (contains '_test' but not 'metahuman')
            elif '_test' in folder and 'metahuman' not in folder:
                test_dirs.append(folder_path)

    # Include metahuman directories in training
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path) and '_metahuman' in folder:
            train_dirs.append(folder_path)

    # Log and count images in train and test directories
    logging.info("Counting images in train and test directories...")
    for train_dir in train_dirs:
        image_count = count_images(train_dir)
        logging.info(f"Train folder {os.path.basename(train_dir)}: {image_count} images")

    for test_dir in test_dirs:
        image_count = count_images(test_dir)
        logging.info(f"Test folder {os.path.basename(test_dir)}: {image_count} images")

    # Process train directories
    train_img_dirs = []
    for train_dir in train_dirs:
        train_img_dirs.extend(glob.glob(f"{train_dir}/*/"))

    if not train_img_dirs:  # If no subfolders, use the main directory
        train_img_dirs = train_dirs

    logging.info(f"Processed train directories: {train_img_dirs}")

    # Initialize datasets
    eyedataset_train = EyeglassDataset(
        image_dir=train_img_dirs,
        augment=True
    )

    eyedataset_val = EyeglassDataset(
        image_dir=test_dirs,
        augment=False
    )

    if not train_dirs:
        raise ValueError("No valid train directories found.")
    if not test_dirs:
        raise ValueError("No valid test directories found.")

    # Create train_loader and val_loader
    train_loader = DataLoader(eyedataset_train, shuffle=True, **loader_args)
    val_loader = DataLoader(eyedataset_val, shuffle=False, **loader_args)

    n_train = len(eyedataset_train)
    n_val = len(eyedataset_val)

    # Initialize wandb
    experiment = wandb.init(project='U-Net-eyeglasses')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    # Log training information
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

    # Set up optimizer, loss, scheduler, and scaler
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)

    # Custom learning rate schedule example
    def adjust_learning_rate(optimizer, epoch):
        if epoch == 2:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 5
        elif epoch == 4:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 1.2

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    global_step = 0

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        train_dice = torch.tensor(0.0, device=device) # modificare

        # Adjust learning rate at specific epochs
        adjust_learning_rate(optimizer, epoch)

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, (
                    f'Network has been defined with {model.n_channels} input channels, ' 
                    f'but loaded images have {images.shape[1]} channels.')

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

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                # Log metrics
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # Save checkpoint
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = eyedataset_train.mask_values
            torch.save(state_dict, str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
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

    # Initialize the U-Net model
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
        logging.error('Detected OutOfMemoryError! '\
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '\
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr/5,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
