# Two arguments are needed to load image files: one is the root of directory,
#                                               the other is a txt file containing filepath and label.
# The created ImageFile object can be passed to a pytorch DataLoader for multi-threading process.
#
# Author : Sen Jia 
#

import torch.utils.data as data

from PIL import Image
from PIL import ImageFilter
import os
import os.path
import numpy as np
import scipy.misc as misc
import torch
import torchvision.transforms as transforms

from random import randint
import random



def make_dataset(root,txt_file):
    images = []
    with open(txt_file,"r") as f:
        
        for line in f:
            strs = line.rstrip("\n").split(",")
          
            images.append(( os.path.join(root, 'Saliencyresize-total',strs[0]), os.path.join(root, 'HeatmapGT', strs[1]), os.path.join(root, 'HeatmapGT', strs[2]), os.path.join(root, 'HeatmapGT', strs[3]), os.path.join(root,'HeatmapGT', strs[4]) ))
            
            print(images[0])
    return images

def pil_loader(path):
    # print  (path)
    return Image.open(path).convert('RGB')

def map_loader(path):
    # print(path)
    return Image.open(path).convert('L')

def accimage_loader(path):
    import accimage
    try:

        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):

    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageList(data.Dataset):

    def __init__(self, root, txt_file, transform=None, target_transform=None, map_size=None,
                 loader=default_loader, map_loader=map_loader, size_out=None, aug=False):
        imgs = make_dataset(root, txt_file)
        if not imgs:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.map_loader = map_loader
        self.mid = False

    def __getitem__(self, index):
        
        def _post_process(smap):
            smap = smap - smap.min()
            smap = smap / smap.max()
            return smap

        # img_path, fix_path, map_path = self.imgs[index]
        # import ipdb; ipdb.set_trace()

        img_path, map_path_0, map_path_1, map_path_2, map_path_3 = self.imgs[index]

        imgIndex = img_path.split('/')[-1]
        nameDir = img_path.split('/')[-2]
        ImgName = nameDir + '-' + imgIndex


        img = self.loader(os.path.join(img_path))
        w, h = img.size

        s_map_0 = self.map_loader(map_path_0)
        s_map_1 = self.map_loader(map_path_1)
        s_map_2 = self.map_loader(map_path_2)
        s_map_3 = self.map_loader(map_path_3)


        if self.transform is not None:
            img = self.transform(img)

            s_map_0 = self.transform(s_map_0)
            s_map_1 = self.transform(s_map_1)
            s_map_2 = self.transform(s_map_2)
            s_map_3 = self.transform(s_map_3)

            HeatmapCat=torch.cat((s_map_0, s_map_1, s_map_2, s_map_3), 0)


        return img, HeatmapCat, ImgName
        

    def __len__(self):
        return len(self.imgs)

