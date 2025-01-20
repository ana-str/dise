import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn

from utils.dice_score import multiclass_dice_coeff, dice_coeff
criterion = nn.BCEWithLogitsLoss()


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    total_val_loss = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            # Calculate loss
            if net.n_classes == 1:
                mask_true = mask_true.float()  # For binary classification
                loss = criterion(mask_pred.squeeze(1), mask_true)
            else:
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                loss = criterion(mask_pred, mask_true)

            total_val_loss += loss.item()

            # Calculate Dice score
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)


    average_dice_score = dice_score / max(num_val_batches, 1)
    average_val_loss = total_val_loss / max(num_val_batches, 1)

    net.train()
    return average_dice_score, average_val_loss