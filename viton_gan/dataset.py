import os
import numpy as np
import json
import random
from PIL import Image
from PIL import ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class DatasetBase(Dataset):
    """Base dataset for VITON-GAN.
    """
    def __init__(self, opt, mode, data_list, train=True):
        super(DatasetBase, self).__init__()
        self.data_path = os.path.join(opt.data_root, mode)
        self.train = train
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.transform = transforms.Compose([
                transforms.ToTensor(), # [0,255] to [0,1]
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [0,1] to [-1,1]
                ]) 
        
        person_names = []
        cloth_names = []
        with open(os.path.join(opt.data_root, data_list), 'r') as f:
            for line in f.readlines():
                person_name, cloth_name = line.strip().split()
                person_names.append(person_name)
                cloth_names.append(cloth_name)

        self.person_names = person_names
        self.cloth_names = cloth_names

    def __len__(self):
        return len(self.person_names)

    def _get_mask_arrays(self, person_parse):
        """Split person_parse array into mask channels
        """
        shape = (person_parse > 0).astype(np.float32)
        head = (person_parse == 1).astype(np.float32) + \
                (person_parse == 2).astype(np.float32) + \
                (person_parse == 4).astype(np.float32) + \
                (person_parse == 13).astype(np.float32) # Hat, Hair, Sunglasses, Face
        head = (head > 0).astype(np.float32)
        cloth = (person_parse == 5).astype(np.float32) + \
                (person_parse == 6).astype(np.float32) + \
                (person_parse == 7).astype(np.float32) # Upper-clothes, Dress, Coat
        cloth = (cloth > 0).astype(np.float32)
        body = (person_parse == 1).astype(np.float32) + \
                (person_parse == 2).astype(np.float32) + \
                (person_parse == 3).astype(np.float32) + \
                (person_parse == 4).astype(np.float32) + \
                (person_parse > 7).astype(np.float32) # Neither cloth nor background
        body = (body > 0).astype(np.float32)
        return shape, head, cloth, body # [0,1]

    def _downsample(self, im):
        im = im.resize((self.fine_width//16, self.fine_height//16), Image.BILINEAR)
        return im.resize((self.fine_width, self.fine_height), Image.BILINEAR) 

    def _load_pose(self, pose_name):
        """Load pose json file
        """
        with open(os.path.join(self.data_path, 'pose', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1,3))
        point_num = pose_data.shape[0]
        feature_pose_tensor = torch.zeros(point_num, self.fine_height, self.fine_width) # 18 channels
        r = self.radius
        pose_im = Image.new('L', (self.fine_width, self.fine_height)) # For visualization
        pose_draw = ImageDraw.Draw(pose_im)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i,0]
            pointy = pose_data[i,1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = self.transform(one_map)
            feature_pose_tensor[i] = one_map[0]
        pose_tensor = self.transform(pose_im) # [-1,1]
        return feature_pose_tensor, pose_tensor

    def _get_item_base(self, index):
        # Person
        person_name = self.person_names[index] 
        person_im = Image.open(os.path.join(self.data_path, 'person', person_name))
        person_tensor = self.transform(person_im) # [-1,1]

        # Person-parse
        parse_name = person_name.replace('.jpg', '.png')
        person_parse = Image.open(os.path.join(self.data_path, 'person-parse', parse_name))
        person_parse = np.array(person_parse) # shape: (256,192,3)
        shape_mask, head_mask, cloth_mask, body_mask = self._get_mask_arrays(person_parse)
        shape_im = Image.fromarray((shape_mask*255).astype(np.uint8))
        feature_shape_tensor = self.transform(self._downsample(shape_im)) # [-1,1]
        head_mask_tensor = torch.from_numpy(head_mask) # [0,1]
        feature_head_tensor = person_tensor * head_mask_tensor - (1 - head_mask_tensor) # [-1,1], fill -1 for other parts
        cloth_mask_tensor = torch.from_numpy(cloth_mask) # [0,1]
        cloth_parse_tensor = person_tensor * cloth_mask_tensor + (1 - cloth_mask_tensor) # [-1,1], fill 1 for other parts
        body_mask_tensor = torch.from_numpy(body_mask).unsqueeze(0) # Tensor [0,1]
        
        # Pose keypoints
        pose_name = person_name.replace('.jpg', '_keypoints.json')
        feature_pose_tensor, pose_tensor = self._load_pose(pose_name)
        # Cloth-agnostic representation
        feature_tensor = torch.cat([feature_shape_tensor, feature_head_tensor, feature_pose_tensor], 0) 

        data = {
            'person_name': person_name,    # For visualization or ground truth
            'person': person_tensor, # For visualization or ground truth
            'feature': feature_tensor,   # For input
            'pose': pose_tensor, # For visualization
            'head': feature_head_tensor, # For visualization
            'shape': feature_shape_tensor, # For visualization
            'cloth_parse':     cloth_parse_tensor,   # For ground truth
            'body_mask': body_mask_tensor     # For ground truth
            }

        return data 

def binarized_tensor(arr):
    mask = (arr >= 128).astype(np.float32)
    return torch.from_numpy(mask).unsqueeze(0) # [0,1]

def random_horizontal_flip(data):
    rand = random.random()
    if rand < 0.5:
        return data
    else:
        for key, value in data.items():
            if 'name' in key:
                continue
            else:
                data[key] = torch.flip(value, [2]) # 2 for width
    return data

class GMMDataset(DatasetBase):
    def __getitem__(self, index):
        cloth_name = self.cloth_names[index]
        cloth_im = Image.open(os.path.join(self.data_path, 'cloth', cloth_name))
        cloth_tensor = self.transform(cloth_im)  # [-1,1]
        cloth_mask_im = Image.open(os.path.join(self.data_path, 'cloth-mask', cloth_name))
        cloth_mask_tensor = binarized_tensor(np.array(cloth_mask_im))
        grid_im = Image.open('grid.png')
        grid_tensor = self.transform(grid_im)

        data = self._get_item_base(index)
        data['cloth_name'] = cloth_name # For visualization or input
        data['cloth'] = cloth_tensor # For visualization or input
        data['cloth_mask'] = cloth_mask_tensor # For input
        data['grid'] = grid_tensor # For visualization
        if self.train:
            data = random_horizontal_flip(data) # Data augmentation

        return data

class TOMDataset(DatasetBase):
    def __getitem__(self, index):
        cloth_name = self.cloth_names[index]
        cloth_im = Image.open(os.path.join(self.data_path, 'warp-cloth', cloth_name))
        cloth_tensor = self.transform(cloth_im)  # [-1,1]
        cloth_mask_im = Image.open(os.path.join(self.data_path, 'warp-cloth-mask', cloth_name))
        cloth_mask_tensor = binarized_tensor(np.array(cloth_mask_im))

        data = self._get_item_base(index)
        data['cloth_name'] = cloth_name # For visualization or input
        data['cloth'] = cloth_tensor # For visualization or input
        data['cloth_mask'] = cloth_mask_tensor # For input
        if self.train:
            data = random_horizontal_flip(data) # Data augmentation

        return data