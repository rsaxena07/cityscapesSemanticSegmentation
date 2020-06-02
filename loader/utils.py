import torch
import matplotlib.pyplot as plt
import numpy as np

colors = [  # [  0,   0,   0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]

label_colours = dict(zip(range(19), colors))

def decode_segmap(mask):
    
    r = mask.copy()
    g = mask.copy()
    b = mask.copy()
    
    for cls in range(0, 19): #change
        r[mask==cls] = label_colours[cls][0]
        g[mask==cls] = label_colours[cls][1]
        b[mask==cls] = label_colours[cls][2]
    
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    
    return rgb


def predictionsFlattened(learner, images, device ):
    with torch.no_grad():
        if device is not None:
            images = images.to(device=device)
        pred = learner(images)
        pred_flat = torch.argmax(pred, dim=1)

    return pred_flat

def printSegmaps(images, labels, preds, h=4):

    n_images = images.shape[0]
    fig, ax = plt.subplots(n_images,3,figsize=(6*h,h*n_images))
    if n_images==1:
        res = decode_segmap(preds.cpu().numpy()[0])
        ax[0].imshow(res)
        ax[1].imshow(np.transpose(images[0].cpu(), [1,2,0]))
        ax[2].imshow(decode_segmap(labels.cpu().numpy()[0]))

        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
    else:
        for i in range(n_images):
            res = decode_segmap(preds.cpu().numpy()[i])
            ax[i][0].imshow(res)
            ax[i][1].imshow(np.transpose(images[i].cpu(), [1,2,0]))
            ax[i][2].imshow(decode_segmap(labels.cpu().numpy()[i]))

            ax[i][0].axis('off')
            ax[i][1].axis('off')
            ax[i][2].axis('off')

    plt.show()
    return