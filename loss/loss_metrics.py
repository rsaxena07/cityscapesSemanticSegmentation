# This python implements the custom loss functions and other performance metrics

import torch
import torch.nn as nn
import numpy as np

def accuracy(preds, lbls, ignore_class=None):
    
    pred = preds.cpu()
    lbl = lbls.cpu()
    
    if pred.shape!=lbl.shape:
        return -1  #incorrect shape
    
    if ignore_class is not None:
        unlabelled = lbl==ignore_class
        unlabelled_pixels = np.count_nonzero(unlabelled)
        del unlabelled
    else:
        unlabelled_pixels = 0
    
    total_pixels=1
    #calculate total pixels in batch
    for i in range(len(lbl.shape)):
        total_pixels *= lbl.shape[i]
        
    total_pixels -= unlabelled_pixels 
    
    # calculate pixels classified correctly
    correct_preds = lbl==pred
    correct_pixels = np.count_nonzero(correct_preds)
    
    del pred, lbl, correct_preds
    return correct_pixels/total_pixels

# source thread: https://discuss.pytorch.org/t/one-hot-encoding-with-autograd-dice-loss/9781/5
# preds - NxCxHxW
# target - NxHxW
def dice_coeff_per_channel(pred, target, ignore_index=None, smooth=1):
    
    # encoded_target will store one hot encoded target
    # useful documentation: https://pytorch.org/docs/stable/tensors.html#torch.Tensor.scatter_
    
    encoded_target = pred.detach() * 0
    if ignore_index is not None:
        mask = target == ignore_index
        target = target.clone() 
        target[mask] = 0
        encoded_target.scatter_(1, target.unsqueeze(1), 1)
        mask = mask.unsqueeze(1).expand_as(encoded_target)
        encoded_target[mask] = 0
    else:
        encoded_target.scatter_(1, target.unsqueeze(1), 1)
#     encoded_target = Variable(encoded_target)

    # to make pred one hot encoded
    predFlat= pred.argmax(dim=1)
    pred = pred * 0
    pred.scatter_(1, predFlat.unsqueeze(1), 1)
    del predFlat
    
    # calculate numerator which represents intersection
    numerator = pred*encoded_target
    numerator = 2. * numerator.sum(0).sum(1).sum(1) # sum over samples(N), then HxW
    
    # calculte denominator which sums preds and encoded to represent union
    denominator = pred + encoded_target
    if ignore_index is not None:
        denominator[mask] = 0
    denominator = denominator.sum(0).sum(1).sum(1) 
    # sum over N, then HxW .. adding 1 for special case when deno is 0
    
    numerator+=smooth
    denominator+=smooth
    
    return (numerator/denominator)


def dice_loss_per_channel(pred, target, weights=None, ignore_index=None, smooth=1):
    
    dice_coeff = dice_coeff_per_channel(pred, target, ignore_index=ignore_index, smooth=smooth)
    
    if weights is None:
        weights = 1
    
    loss_per_channel = weights * (1 - dice_coeff)
    
    return loss_per_channel


# wrapper method to be used for loss
def dice_loss(pred, target, weights=None, ignore_index=None, smooth=1, squishPreds=False):
    loss_per_channel = dice_loss_per_channel(pred, target, weights=weights, 
                                             ignore_index=ignore_index, smooth=smooth, squishPreds=squishPreds)
    
    return loss_per_channel.mean()