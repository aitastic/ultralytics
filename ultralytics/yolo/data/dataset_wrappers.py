# Ultralytics YOLO 🚀, AGPL-3.0 license
import random

import collections
from copy import deepcopy
from pathlib import Path
from glob import glob
from typing import Literal

import albumentations as A
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .augment import LetterBox
from .utils import HELP_URL, LOGGER, get_hash, img2label_paths, verify_image_label


class CopyPasteDataset:
    """
    Supplementary Dataset to be used with the Copy-Paste mechanism.
    Loads images and corresponding mask images, as well as annotations.

    Attributes:
        dataset: The base dataset.
        imgsz: The size of the images in the dataset.
    """

    def init_suppl_dataset(self, dataset_path: Path | str) -> Dataset:
        """
        Builds a dataset for copy-pasting from the given path.
        The directory should contain the images and masks in the format 
        Object_ID/$NR_{img,mask}.png
        
        dataset_path
         |
         +- Chair_1
            +- 1_img.png
            +- 1_mask.png
            +- 2_img.png
            +- 2_mask.png
            +- ...
         |
         +- Table_2
            +- 1_img.png
            +- 1_mask.png
            +- 2_img.png
            +- 2_mask.png
            +- ...
         |
         +- Car_3

        or
        
        {class_name}_{class_id}_{nr}_{'mask'|'rgb'}.png
        # TODO decide which one


        Attributes:
            dataset_path: The path to the images and masks
        """
        # get images from directory
        images = glob(f'{dataset_path}/**_rgb.png')
        objects = {}
        for img_path in images:
            class_name, class_id, nr, img_type = img_path.rstrip('.png').split('/')[-1].rsplit('_', maxsplit=3)
            if not class_name in objects:
                objects[class_name] = {}
            objects[class_name].update(
                class_id=class_id,
            )
            if not 'paths' in objects[class_name]:
                objects[class_name]['paths'] = []
            objects[class_name]['paths'].append(img_path)
        return objects


    def __init__(
        self,
        base_dataset,
        suppl_dataset_path: str | Path,
        split: Literal['train', 'val', 'test'],
        augmentations: dict,
        p: float=0.6,
        max_pasted_objects: int=3
    ):
        """
        Args:
            dataset (BaseDataset): The base dataset to apply transformations to.
            suppl_dataset_path (Path | str): Path to the directory containing the supplementary dataset.
                                             This contains images as well as masks used to cut out the objects.
            split (str): Which dataset split to use ('train', 'val', 'test')
            augmentations (dict): A mapping of augmentations to use and their respective chance
            p (float): Probability that an object is pasted onto an image
            max_pasted_objects (int): Maximum number of objects that are pasted onto an image
        """
        self.dataset = base_dataset
        self.imgsz = base_dataset.imgsz
        self.batch_size = base_dataset.batch_size
        self.split = split
        self.augmentations = augmentations
        self.p = p
        self.max_pasted_objects = max_pasted_objects
        self.batch_idx = 0
        self.suppl_dataset = self.init_suppl_dataset(Path(suppl_dataset_path) / split)
        self.labels = self.dataset.labels
        self.im_files = self.dataset.im_files
        self.cache_labels = self.dataset.cache_labels

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.dataset)

    def convert_tensor_to_cv(self, img: torch.Tensor) -> np.ndarray:
        img = img.permute(1,2,0).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def convert_cv_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = torch.tensor(img).permute(2,0,1)
        return img

    def _vis(self, labels, debug_view=False):
        if debug_view:
            # Load image, transpose to cv2 format, reorder color channels and normalize
            img = self.convert_tensor_to_cv(labels['img'])
            for bbox, class_id in zip(labels['bboxes'], labels['cls']):
                center_x, center_y, width, height = bbox

                # Convert YOLO bounding box coordinates to OpenCV format
                center_x *= img.shape[1]
                center_y *= img.shape[0]
                width *= img.shape[1]
                height *= img.shape[0]
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                # Draw the bounding box
                cv2.rectangle(img, (x, y), (x + int(width), y + int(height)), (0, 255, 0), 2)

                # Draw the class label
                cv2.putText(img, str(class_id[0]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            # Show image
            cv2.imshow('debug', img)
            cv2.waitKey()

    def copy_paste(self, labels, suppl_img_path, suppl_mask_path, suppl_obj_id, index):
        # Load images
        suppl_img = cv2.imread(suppl_img_path)
        suppl_mask = cv2.imread(suppl_mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize to be inline with the base images
        suppl_img = cv2.resize(suppl_img, (self.imgsz, self.imgsz))
        suppl_mask = cv2.resize(suppl_mask, (self.imgsz, self.imgsz))

        # Fix channel order
        base_img = self.convert_tensor_to_cv(labels['img'])

        h,w = suppl_img.shape[:2]
        
        # Convert the mask image to binary
        _, binary_mask = cv2.threshold(suppl_mask, 45, 255, cv2.THRESH_BINARY)
        
        # Ensure the single-channel mask is of type uint8
        binary_mask = binary_mask.astype(np.uint8)

        # Find the bounding box of the object
        x, y, w, h = cv2.boundingRect(binary_mask)
        
        # Extract the object
        obj = cv2.bitwise_and(suppl_img, suppl_img, mask=binary_mask)
        
        # Cut out the object and its mask
        obj = obj[y:y+h, x:x+w]
        binary_mask= binary_mask[y:y+h, x:x+w]
        
        # Select a random position in the second image
        x2 = max(0, np.random.randint(-w + 1, base_img.shape[1]))
        y2 = max(0, np.random.randint(-h + 1, base_img.shape[0]))
        
        # Calculate the width and height of the part of the object that fits within the second image
        w2 = min(w, base_img.shape[1] - x2)
        h2 = min(h, base_img.shape[0] - y2)
        
        # Adjust the object and its mask to fit within the second image
        obj = obj[:h2, :w2]
        binary_mask = binary_mask[:h2, :w2]

        # Paste the object onto the second image using the mask
        base_img[y2:y2+h2, x2:x2+w2] = cv2.bitwise_and(
            base_img[y2:y2+h2, x2:x2+w2],
            base_img[y2:y2+h2, x2:x2+w2],
            mask=cv2.bitwise_not(binary_mask)
        )
        base_img[y2:y2+h2, x2:x2+w2] = cv2.bitwise_or(
            base_img[y2:y2+h2, x2:x2+w2],
            obj
        )

        # Calculate YOLO annotations
        obj_class_id = int(suppl_obj_id) - 77        # Just for now to align with expected values
        obj_bbox_center_x = (x2+w2/2) / base_img.shape[1]
        obj_bbox_center_y = (y2+h2/2) / base_img.shape[0]
        obj_bbox_w = w2 / base_img.shape[1] 
        obj_bbox_h = h2 / base_img.shape[0]

        # Create a new bounding box
        new_bbox = torch.tensor([[obj_bbox_center_x, obj_bbox_center_y, obj_bbox_w, obj_bbox_h]])

        # Add the new bounding box to the labels
        if labels['bboxes'].shape == (1,1,0) or torch.equal(labels['bboxes'], torch.tensor([[-1, -1, -1, -1]])):
            # Create new tensors
            labels['cls'] = torch.tensor([[obj_class_id]])
            labels['batch_idx'] = torch.tensor([index])
            labels['bboxes'] = new_bbox
        else:
            # Append to labels
            labels['cls'] = torch.cat((labels['cls'], torch.tensor([[obj_class_id]])))
            labels['batch_idx'] = torch.cat((labels['batch_idx'], torch.tensor([index])))    # really?
            labels['bboxes'] = torch.cat((labels['bboxes'], new_bbox))

        # print(f"{(labels['cls'], labels['batch_idx'], labels['bboxes'])=}")

        # Cast image back to tensor
        labels['img'] = self.convert_cv_to_tensor(base_img)
            
        self._vis(labels)
        
        return labels

    def _select_object(self, dataset):
        """
        select random object from our supplementary dataset
        """
        obj = random.sample(sorted(dataset), k=1)[0]
        obj_id = self.suppl_dataset[obj]['class_id']
        img = random.sample(sorted(dataset[obj]['paths']), k=1)[0]
        mask = img.replace('rgb', 'mask')
        return obj, obj_id, img, mask
        

    def augment(self, labels: dict):
        if torch.equal(labels['bboxes'], torch.tensor([[-1, -1, -1, -1]])):
            # skip if no objects in image
            return labels

        transforms = []
        for augmentation, chance in self.augmentations.items():
            if augmentation == 'random_crop':
                w, h = labels['img'].shape[1:]
                size_factor = random.uniform(0.3, 0.9)
                width = min(int(size_factor * self.imgsz), w)
                height = min(int(size_factor * self.imgsz), h)
                transforms.append(
                        A.RandomCrop(p=chance or 0.5, width=width, height=height),
                        )
            if augmentation == 'horizontal_flip':
                transforms.append(
                        A.HorizontalFlip(p=chance or 0.5),
                        )
            if augmentation == 'random_brightness_contrast':
                transforms.append(
                        A.RandomBrightnessContrast(p=chance or 0.2),
                        )
            if augmentation == 'channel_shuffle':
                transforms.append(
                        A.ChannelShuffle(p=chance or 0.6)
                        )
            if augmentation == 'color_jitter':
                transforms.append(
                        A.ColorJitter(p=chance or 0.7)
                        )
            if augmentation == 'hsv_shift':
                transforms.append(
                        A.HueSaturationValue(p=chance or 0.7)
                        )
            if augmentation == 'gauss_noise':
                transforms.append(
                        A.GaussNoise(p=chance or 0.7)
                        )
            if augmentation == 'iso_noise':
                transforms.append(
                        A.ISONoise(p=chance or 0.7)
                        )
            if augmentation == 'image_compression':
                transforms.append(
                        A.ImageCompression(p=chance or 0.5)
                        )


        # always pad if needed
        transforms.append(
                A.PadIfNeeded(min_height=self.imgsz, min_width=self.imgsz, p=1, border_mode=cv2.BORDER_CONSTANT)
                )

        transforms = A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                min_visibility=0.1,
                label_fields=['class_labels'],
            )
        )
        img = self.convert_tensor_to_cv(labels['img'])
        transformed = transforms(
            image=img,
            bboxes=labels['bboxes'],
            class_labels=labels['cls']
        )
        if not transformed['class_labels']:
            # reset to empty
            labels.update({
                'cls': torch.tensor([[-1]]),
                'batch_idx': torch.tensor([-1]),
                'bboxes': torch.tensor([[-1, -1, -1, -1]]),
                })
            return labels

        labels['cls'] = torch.stack(transformed['class_labels'])

        labels['img'] = self.convert_cv_to_tensor(transformed['image'])
        labels['bboxes'] = torch.tensor(transformed['bboxes'])
        labels['cls'] = torch.stack(transformed['class_labels'])

        # Strip away batch indices for classes that got lost during augmentation
        labels['batch_idx'] = labels['batch_idx'][:len(labels['cls'])]

        return labels

    def __getitem__(self, index):
        """
        Applies CopyPaste to an item in the dataset.

        Args:
            index (int): Index of the item in the dataset.

        Returns:
            (dict): A dictionary containing the transformed item data.
        """
        labels = deepcopy(self.dataset[index])
        
        # empty original labels, we need to get rid of them
        labels.update({
            'cls': torch.tensor([[-1]]),
            'batch_idx': torch.tensor([self.batch_idx]),
            'bboxes': torch.tensor([[-1, -1, -1, -1]]),
            })

        for _ in range(self.max_pasted_objects):
            if random.uniform(0, 1) < self.p:
                _, obj_id, suppl_img, suppl_mask = self._select_object(self.suppl_dataset)
                labels = self.copy_paste(labels, suppl_img, suppl_mask, obj_id, self.batch_idx)

        if self.augmentations:
            labels = self.augment(labels)

        self.batch_idx += 1
        self.batch_idx %= self.batch_size
        
        return labels

    def collate_fn(self, batch):
        # Initialize lists to store the batched data
        imgs = []
        cls = []
        bboxes = []
        batch_idx = []
        im_files = []
        ori_shapes = []
        resized_shapes = []

        # Iterate over each item in the batch
        for item in batch:
            # Append the data to the corresponding lists
            imgs.append(item['img'])
            cls.append(item['cls'])
            bboxes.append(item['bboxes'])
            batch_idx.append(item['batch_idx'])
            im_files.append(item['im_file'])
            ori_shapes.append(item['ori_shape'])
            resized_shapes.append(item['resized_shape'])

        # Stack the images into a single tensor
        imgs = torch.stack(imgs)

        # Convert the lists of labels and bounding boxes to padded tensors
        cls = pad_sequence(cls, batch_first=True, padding_value=-1)
        bboxes = pad_sequence(bboxes, batch_first=True, padding_value=-1)
        batch_idx = pad_sequence(batch_idx, batch_first=True, padding_value=-1)
    
        # Return the batched data
        result = {
            'img': imgs,
            'cls': cls,
            'bboxes': bboxes,
            'batch_idx': batch_idx,
            'im_file': im_files,
            'ori_shape': ori_shapes,
            'resized_shape': resized_shapes,
        }
        return result


class MixAndRectDataset:
    """
    A dataset class that applies mosaic and mixup transformations as well as rectangular training.

    Attributes:
        dataset: The base dataset.
        imgsz: The size of the images in the dataset.
    """

    def __init__(self, dataset):
        """
        Args:
            dataset (BaseDataset): The base dataset to apply transformations to.
        """
        self.dataset = dataset
        self.imgsz = dataset.imgsz

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Applies mosaic, mixup and rectangular training transformations to an item in the dataset.

        Args:
            index (int): Index of the item in the dataset.

        Returns:
            (dict): A dictionary containing the transformed item data.
        """
        labels = deepcopy(self.dataset[index])
        for transform in self.dataset.transforms.tolist():
            # Mosaic and mixup
            if hasattr(transform, 'get_indexes'):
                indexes = transform.get_indexes(self.dataset)
                if not isinstance(indexes, collections.abc.Sequence):
                    indexes = [indexes]
                labels['mix_labels'] = [deepcopy(self.dataset[index]) for index in indexes]
            if self.dataset.rect and isinstance(transform, LetterBox):
                transform.new_shape = self.dataset.batch_shapes[self.dataset.batch[index]]
            labels = transform(labels)
            if 'mix_labels' in labels:
                labels.pop('mix_labels')
        return labels
