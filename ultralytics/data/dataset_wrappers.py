# Ultralytics YOLO 🚀, AGPL-3.0 license
import random

import collections
from collections import defaultdict
from copy import deepcopy
from typing import Literal

import albumentations as A
import cv2
import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence
from ultralytics.data import BaseDataset


class CopyPasteDataset:
    """
    Supplementary Dataset to be used with the Copy-Paste mechanism.
    Loads images and corresponding mask images, as well as annotations.

    Attributes:
        dataset: The base dataset.
        imgsz: The size of the images in the dataset.
    """
    def __init__(
            self,
            base_dataset: BaseDataset,
            suppl_dataset: BaseDataset,
            split: Literal['train', 'val', 'test'],
            augmentations: dict,
            p: float=0.6,
            max_pasted_objects: int=3
            ):
        """
        Args:
            dataset (BaseDataset): The base dataset to apply transformations to.
            suppl_dataset (BaseDataset): Supplementary dataset containing labels and images, as well as masks used to cut out the objects.
            split (str): Which dataset split to use ('train', 'val', 'test')
            object_augmentations (dict): A mapping of augmentations to use on the object level, and their respective chance
            image_augmentations (dict): A mapping of augmentations to use on the entire image, and their respective chance
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
        self.suppl_dataset = suppl_dataset
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
        if debug_view and 'bboxes' in labels:
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

    def binary_mask_from_seg_labels(self, image_shape, mask_labels):
        # Create a blank mask image of the same size
        mask = np.zeros(image_shape[:2], dtype=np.uint8)

        # Assume each segment is separated by a special marker (e.g., [None, None])
        segments = []
        segment = []
        for label in mask_labels:
            if label == [None, None]:  # Change this to your actual separator
                segments.append(segment)
                segment = []
            else:
                segment.append(label)
        if segment:
            segments.append(segment)

        for segment in segments:
            points = np.array(segment, dtype=float).reshape(-1, 2)
            # Convert the points to the original image scale
            points = (points * np.array([image_shape[1], image_shape[0]])).astype(np.int32)
            # Draw the contour on the mask image
            cv2.drawContours(mask, [points], 0, (255), -1)  # filled contours

        return mask


    def copy_paste(
            self,
            labels: dict,
            suppl_labels: dict,
            index: int,
            overlap_threshold: float = 0.5
            ):

        # Load images
        suppl_img = self.convert_tensor_to_cv(suppl_labels['img'])
        base_img = self.convert_tensor_to_cv(labels['img'])

        # Resize to be inline with the base images
        suppl_img = cv2.resize(suppl_img, (self.imgsz, self.imgsz))

        # FIXME during validation this is required. I'm sceptical that's correct.
        base_img = cv2.resize(base_img, (self.imgsz, self.imgsz))

        # Get image shape
        h,w = suppl_img.shape[:2]

        # Convert the mask image to binary
        suppl_mask = suppl_labels['masks'][0].numpy() * 255

        # Ensure mask is same size as image
        suppl_mask = cv2.resize(suppl_mask, (self.imgsz, self.imgsz))

        # Convert to single channel if necessary
        if suppl_mask.shape[-1] == 3:
            suppl_mask = cv2.cvtColor(suppl_mask, cv2.COLOR_BGR2GRAY)
        
        # Convert the mask image to binary
        _, binary_mask = cv2.threshold(suppl_mask, 1, 255, cv2.THRESH_BINARY)
        
        # Ensure the single-channel mask is of type uint8
        binary_mask = binary_mask.astype(np.uint8)

        # Find the bounding box of the object
        x, y, w, h = cv2.boundingRect(binary_mask)

        # Extract the object
        obj = cv2.bitwise_and(suppl_img, suppl_img, mask=binary_mask)

        # Cut out the object and its mask
        obj = obj[y:y+h, x:x+w]
        binary_mask = binary_mask[y:y+h, x:x+w]

        # Initialize object mask
        segmentation_mask = np.zeros_like(base_img[:,:, 0], dtype=np.uint8)

        # Initialize occupancy mask if it doesn't exist
        occupancy_mask = labels.get('occupancy_mask', segmentation_mask) 

        for _ in range(512):
            # Select a random position in the second image
            x2 = max(0, np.random.randint(-w + 1, base_img.shape[1]))
            y2 = max(0, np.random.randint(-h + 1, base_img.shape[0]))

            # Check if the selected area is too occupied
            if np.mean(occupancy_mask[y2:y2+h, x2:x2+w]) < overlap_threshold:
                break
        else:
            # Could not find a suitable free spot
            # Returning without adding object
            return labels

        # Update the occupancy mask
        w2 = min(w, occupancy_mask.shape[1] - x2)
        h2 = min(h, occupancy_mask.shape[0] - y2)

        occupancy_mask[y2:y2+h, x2:x2+w] = binary_mask[:h2, :w2]
        segmentation_mask[y2:y2+h, x2:x2+w] = binary_mask[:h2, :w2]

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
        obj_class_id = suppl_labels['cls']
        obj_bbox_center_x = (x2+w2/2) / base_img.shape[1]
        obj_bbox_center_y = (y2+h2/2) / base_img.shape[0]
        obj_bbox_w = w2 / base_img.shape[1] 
        obj_bbox_h = h2 / base_img.shape[0]

        # Apparently sometimes the resulting bboxes are 0-width, which causes issues.
        # If that happens, just skip it
        if obj_bbox_w < 0.01 or obj_bbox_h < 0.01:
            return labels

        # Create a new bounding box
        new_bbox = torch.tensor([[obj_bbox_center_x, obj_bbox_center_y, obj_bbox_w, obj_bbox_h]])

        # Add the new bounding box to the labels
        if 'bboxes' not in labels:
            # Create new tensors
            labels['cls'] = torch.tensor([[obj_class_id]])
            labels['batch_idx'] = torch.tensor([index])
            labels['bboxes'] = new_bbox
            labels['masks'] = [segmentation_mask]
        else:
            # Append to labels
            labels['cls'] = torch.cat((labels['cls'], torch.tensor([[obj_class_id]])))
            labels['batch_idx'] = torch.cat((labels['batch_idx'], torch.tensor([index])))
            labels['bboxes'] = torch.cat((labels['bboxes'], new_bbox))
            labels['masks'].append(segmentation_mask.copy())

        # Cast image back to tensor
        labels['img'] = self.convert_cv_to_tensor(base_img)

        # Store occupancy_mask
        labels['occupancy_mask'] = occupancy_mask

        self._vis(labels)

        return labels

    def augment(self, labels: dict, augmentations: dict, min_visible_pixels: int = 8096):
        transform_dict = defaultdict(list)
        for augmentation, chance in augmentations.items():
            if augmentation == 'random_crop':
                w, h = labels['img'].shape[1:]
                size_factor = random.uniform(0.3, 0.9)
                width = min(int(size_factor * self.imgsz), w)
                height = min(int(size_factor * self.imgsz), h)
                transform_dict['crop'].append(
                        A.RandomCrop(
                            p=chance or 0.5,
                            width=width,
                            height=height
                            ),
                        )
            if augmentation == 'advanced_blur':
                transform_dict['blur'].append(
                        A.AdvancedBlur(p=chance or 0.2),
                        )
            if augmentation == 'horizontal_flip':
                transform_dict['geometric'].append(
                        A.HorizontalFlip(p=chance or 0.5),
                        )
            if augmentation == 'random_brightness_contrast':
                transform_dict['photometric'].append( 
                        A.RandomBrightnessContrast(p=chance or 0.2),
                        )
            if augmentation == 'channel_shuffle':
                transform_dict['photometric'].append(
                        A.ChannelShuffle(p=chance or 0.4)
                        )
            if augmentation == 'color_jitter':
                transform_dict['photometric'].append(
                        A.ColorJitter(
                            p=chance or 0.4,
                            brightness=0.3,
                            contrast=0.3,
                            saturation=0.3,
                            )
                        )
            if augmentation == 'hsv_shift':
                transform_dict['photometric'].append(
                        A.HueSaturationValue(
                            p=chance or 0.4,
                            hue_shift_limit=(-50, 50),
                            sat_shift_limit=(-50, 50),
                            val_shift_limit=(-50, 50),
                            )
                        )
            if augmentation == 'gauss_noise':
                transform_dict['noise'].append(
                        A.GaussNoise(p=chance or 0.7)
                        )
            if augmentation == 'iso_noise':
                transform_dict['noise'].append(
                        A.ISONoise(p=chance or 0.7)
                        )
            if augmentation == 'image_compression':
                transform_dict['noise'].append(
                        A.ImageCompression(p=chance or 0.5)
                        )

        transforms = []
        # Limit transforms for performance reasons
        for category in transform_dict.keys():
            if category != 'photometric':
                # Only choose one transform per category
                transforms.extend(random.sample(transform_dict[category], k=1))
            else:
                # We want to combine the photometric transforms, so choose multiples
                transforms.extend(random.sample(
                    transform_dict[category],
                    k=min(3, len(transform_dict[category]))
                ))

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
                    ),
                )

        # Convert image to cv format
        img = self.convert_tensor_to_cv(labels['img'])
        
        # Make sure that the masks are lists of numpy arrays
        masks = labels.get('masks')
        if isinstance(masks, torch.Tensor):
            masks = [self.convert_tensor_to_cv(masks)]


        # BBox width/height > 1. indicates unnormalized coordinates, values <=0 indicate broken boxes.
        bboxes = []
        for bbox in labels.get('bboxes', []):
            boxlen = len(bboxes)
            if np.any([v >= 1. for v in bbox.numpy()]):
                bboxes.append(bbox / self.imgsz)
            if np.any([v <= 0.0 for v in bbox.numpy()]):
                print(f'augment: bboxes <= 0.: {bbox=}')
                print(f'{labels["im_file"]=}')
                bboxes.append(torch.clamp(bbox, 0.001, 1.))
            # check if a box has already been added this iteration
            if boxlen == len(bboxes):
                bboxes.append(bbox)
        labels['bboxes'] = torch.stack(bboxes)

        # Apply augmentations
        transformed = transforms(
                image=img,
                bboxes=labels.get('bboxes', []),
                class_labels=labels.get('cls', []),
                masks=masks,
                )

        # Convert image back to tensor
        labels['img'] = self.convert_cv_to_tensor(transformed['image'])

        if 'masks' not in labels:
            return labels

        # Filter out those objects that have few visible pixels
        instances = [
                (bbox, class_label, torch.tensor(mask))
                for bbox, class_label, mask in zip(
                    transformed['bboxes'],
                    transformed['class_labels'],
                    transformed['masks']
                    )
                if np.count_nonzero(mask) > min_visible_pixels
                ]

        bboxes = [inst[0] for inst in instances]
        class_labels = [inst[1] for inst in instances]
        masks = [inst[2] for inst in instances]

        # don't update anything but image if there are no objects 
        if 'bboxes' not in labels or not bboxes:
            return labels

        # Stack lists back into tensors
        labels['bboxes'] = torch.tensor(bboxes).float()
        labels['cls'] = torch.stack(class_labels)
        labels['masks'] = torch.stack(masks)

        # Strip away batch indices for classes that got lost during augmentation
        labels['batch_idx'] = labels['batch_idx'][:len(class_labels)]

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
        for key in ['cls', 'bboxes', 'batch_idx']:
            labels.pop(key)

        for _ in range(self.max_pasted_objects):
            if random.uniform(0, 1) < self.p:
                # Choose randomly from the supplementary dataset
                suppl_labels = self.suppl_dataset[random.randint(0, len(self.suppl_dataset)-1)]
                if self.augmentations:
                    suppl_labels = self.augment(suppl_labels, augmentations=self.augmentations['object_level'])
                # Combine labels via CopyPaste
                labels = self.copy_paste(labels, suppl_labels, self.batch_idx)
            else:
                # This is required for validation, as images here are potentially of different shapes
                img = self.convert_tensor_to_cv(labels['img'])
                labels['img'] = self.convert_cv_to_tensor(cv2.resize(img, (self.imgsz, self.imgsz)))
                if 'masks' in labels:
                    mask = labels['masks'][-1]
                    labels['masks'][-1] = cv2.resize(mask, (self.imgsz, self.imgsz))
        
        if self.augmentations and 'masks' in labels:
            labels = self.augment(labels, augmentations=self.augmentations['image_level'])

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
            im_files.append(item['im_file'])
            ori_shapes.append(item['ori_shape'])
            resized_shapes.append(item['resized_shape'])
            if 'cls' in item:
                cls.append(item['cls'])
                bboxes.append(item['bboxes'])
                batch_idx.append(item['batch_idx'])
        try:
            # Stack the images into a single tensor
            imgs = torch.stack(imgs)
        except Exception as e:
            print(f'{imgs=}')
            for img, file in zip(imgs, im_files):
                print(f'{file=}: {img.shape=}')

        # Convert the lists of labels and bounding boxes to padded tensors
        cls = pad_sequence(cls, batch_first=True, padding_value=-1)
        bboxes = pad_sequence(bboxes, batch_first=True, padding_value=-1).float()
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

