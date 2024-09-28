import os
from torch.utils.data import Dataset
from PIL import Image
from natsort import natsorted
import numpy as np
from utilities.python_pfm import readPFM
from dataset.preprocess import custom_transform

class ScaredDataset(Dataset):
    def __init__(self, datadir, split='train'):
        """
        datadir: path/to/Scared
        split: train, val, or test        
        """
        super(ScaredDataset, self).__init__()
        self.datadir = datadir
        self.split = split
        self.split_datasets = {
            'train': ['dataset_1', 'dataset_2', 'dataset_3', 'dataset_4', 'dataset_5', 'dataset_6'],
            'validation': ['dataset_7'],
            'test': ['dataset_8']
        }

        self.relevant_folders = self.split_datasets[self.split]

        self._load_img()
        self._transformation()

    def _load_img(self):
        self.left_img = []
        self.right_img = []
        self.disp = []
        self.occl = []

        for dataset in self.relevant_folders:
            dataset_path = os.path.join(self.datadir, dataset)
            keyframes = os.listdir(dataset_path)
            for keyframe in keyframes:
                keyframe_path = os.path.join(dataset_path, keyframe, 'data')

                # Load left images
                left_path = os.path.join(keyframe_path, 'left_finalpass')
                left_images = [os.path.join(left_path, file) for file in os.listdir(left_path) if file.endswith('.png')]
                left_images = natsorted(left_images)
                self.left_img.extend(left_images)

                # Load right images
                right_path = os.path.join(keyframe_path, 'right_finalpass')
                right_images = [os.path.join(right_path, file) for file in os.listdir(right_path) if file.endswith('.png')]
                right_images = natsorted(right_images)
                self.right_img.extend(right_images)

                # Load disparity maps
                disp_path = os.path.join(keyframe_path, 'disparity_left_pfm')
                disp_images = [os.path.join(disp_path, file) for file in os.listdir(disp_path) if file.endswith('.pfm')]
                disp_images = natsorted(disp_images)
                self.disp.extend(disp_images)

                # Load occlusion map
                occl_path = os.path.join(keyframe_path, 'occl_left')
                occl_files = [os.path.join(occl_path, file) for file in os.listdir(occl_path) if file.endswith('.png')]
                occl_files = natsorted(occl_files)
                self.occl.extend(occl_files)

    def _transformation(self):
        self.transformation = None

    def __len__(self):
        return len(self.left_img)
    
    def __getitem__(self, idx):
        inputs = {}

         # Load left and right images
        left_img_fname = self.left_img[idx]
        right_img_fname = self.right_img[idx]
        inputs['left'] = np.array(Image.open(left_img_fname)).astype(np.uint8)
        inputs['right'] = np.array(Image.open(right_img_fname)).astype(np.uint8)

        # disparity maps
        disp_fname = self.disp[idx]
        disparity, _ = readPFM(disp_fname)
        inputs['disp'] = disparity

        # occlution masks
        occ_fname = self.occl[idx]
        inputs['occ_mask'] = np.array(Image.open(occ_fname)).astype(np.uint8) == 128
        inputs['disp'][inputs['occ_mask']] == 0.0

        inputs = custom_transform(inputs, self.transformation)
        return inputs
