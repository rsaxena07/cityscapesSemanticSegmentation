from torchvision.datasets import Cityscapes
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
import numpy as np

class CityscapesDataset(Cityscapes):
    
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

    
    def __init__(self, root, split='train', mode='fine', target_type='semantic', shrinkToSize=None, 
                 cropHeight=512, cropWidth=1024, transform=None, target_transform=None):
        super(CityscapesDataset, self).__init__(root, split=split, mode=mode, target_type=target_type)
        
        #defining constants
        self.n_classes = 19
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [ 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = [
            "road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light", "traffic_sign",
            "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", 
            "bicycle",
        ]
        
        self.ignore_index = -1
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))

        
        #defining transform params
        self.op_size = (cropHeight, cropWidth)
        self.transform = transform
        self.target_transform = target_transform

        
        if shrinkToSize is not None:
            self.images = self.images[:shrinkToSize]
            self.targets = self.targets[:shrinkToSize]
        
        return
    
    def __getitem__(self, index):
        img, lbl = super().__getitem__(index)
        img, lbl = self.transformImgMask(img, lbl)
        
        # remove void classes from label
        lbl = self.encode_segmap(lbl)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            lbl = self.target_transform(lbl)
        
        return img, lbl
        
    def __len__(self):
        return len(self.images)
    
    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
    
    def decode_segmap(mask):
        r = mask.copy()
        g = mask.copy()
        b = mask.copy()

        for cls in range(0, self.n_classes): #change
            r[mask==cls] = self.label_colours[cls][0]
            g[mask==cls] = self.label_colours[cls][1]
            b[mask==cls] = self.label_colours[cls][2]

        rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0

        return rgb
    
    def transformImgMask(self, image, mask):
        
        #crop only training images
        if self.split=='train':
            # Random Crop
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.op_size)
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
        
        # Implement other transforms like vertical and horizontal flips
        
        # Transform to Tensor
        image = self.transformToTensor(image, normalize=True)
        mask = self.transformToTensor(mask, normalize=False)
        return image, mask
    
    def transformToTensor(self, img, normalize=False):
        
        res = torch.from_numpy(np.array(img, np.int64, copy=False))
        res = res.view(img.size[1], img.size[0], len(img.getbands()))
        res = res.permute((2, 0, 1)).contiguous()
        if normalize:
            return res.float().div(255)
        return res