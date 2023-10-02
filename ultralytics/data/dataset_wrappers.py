# Ultralytics YOLO 🚀, AGPL-3.0 license
import random

import copy
import time
from collections import defaultdict
from functools import wraps
from copy import deepcopy
from typing import Literal

import albumentations as A
import cv2
import numpy as np
import torch

from ultralytics.data import BaseDataset


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


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
            max_pasted_objects: int=6
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
        # Use generator to get samples from our suppl_dataset
        self.object_sampler = self._get_objects()

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.dataset)

    def convert_tensor_to_cv(self, img: torch.Tensor) -> np.ndarray:
        if isinstance(img, np.ndarray):
            # Fix channel order, if colors are first
            if len(img.shape) == 3 and img.shape.index(min(img.shape)) == 0:
                img = np.transpose(img, (1, 2, 0))
            return img
        if len(img.shape) == 3:
            img = img.permute(1,2,0).numpy()            # Result: HxWxC
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = img.numpy()
        return img

    def convert_cv_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        if isinstance(img, torch.Tensor):
            return img
        if len(img.shape) < 3:
            return torch.tensor(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = torch.tensor(img).permute(2,0,1)      # Result: CxHxW
        return img

    def _vis(self, labels, debug_view=False, window_title='debug'):
        if debug_view:
            # Load image, transpose to cv2 format, reorder color channels and normalize
            img = self.convert_tensor_to_cv(labels['img'])
            if 'bboxes' in labels and 'cls' in labels:
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
                    try:
                        cv2.putText(img, str(class_id[0]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    except:
                        pass

            # Show image
            cv2.imshow(window_title, img)
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

    def recompute_bboxes(self, labels):
        labels['bboxes'] = torch.zeros([len(labels.get('cls', [])), 4])
        for i, (_, mask) in enumerate(zip(labels.get('cls', []), labels.get('masks', []))):
            mask = self.convert_tensor_to_cv(mask)
            # Find the bounding box of the object
            x, y, w, h = cv2.boundingRect(mask)

            # FIXME take a look at what this should be at all stages
            base_img = self.convert_tensor_to_cv(labels['img'])

            # Calculate YOLO annotations
            obj_bbox_center_x = (x+w/2) / base_img.shape[1]
            obj_bbox_center_y = (y+h/2) / base_img.shape[0]
            obj_bbox_w = w / base_img.shape[1] 
            obj_bbox_h = h / base_img.shape[0]

            labels['bboxes'][i] = torch.tensor([[obj_bbox_center_x, obj_bbox_center_y, obj_bbox_w, obj_bbox_h]])
        return labels


    def feather_insert_object(self, base_img, obj_img, binary_mask, pos, max_feather_amount=31):
        """
        Insert an object into a base image with feathering at the edges for smooth transitions.
        
        :param base_img: The base image where the object is being pasted
        :param obj_img: The object image to be pasted
        :param binary_mask: The binary mask of the object
        :param pos: The position where the object is to be pasted
        :param feather_amount: The width of the feathering effect
        :return: The modified image
        """
        if isinstance(base_img, torch.Tensor):
            base_img = self.convert_tensor_to_cv(base_img)
        if isinstance(obj_img, torch.Tensor):
            obj_img = self.convert_tensor_to_cv(obj_img)
        if isinstance(binary_mask, torch.Tensor):
            binary_mask = self.convert_tensor_to_cv(binary_mask)

        # Randomly choose strength of feathering
        feather_amount = int(random.uniform(0, max_feather_amount))

        # Ensure it's odd
        if feather_amount % 2 == 0:
            feather_amount += 1
        
        # Create a feathering mask
        feather_mask = cv2.GaussianBlur(binary_mask, (feather_amount, feather_amount), 0)
        
        # Normalize the feathering mask to [0, 1]
        alpha_feather = cv2.normalize(feather_mask.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        
        # Replicate the single channel feathering mask into 3 channels
        alpha_colored = cv2.merge((alpha_feather, alpha_feather, alpha_feather))
        
        # Position to insert object
        y, x = pos
        
        # Make sure we are not going out of bounds and the slices are non-empty
        h, w = binary_mask.shape
        y = max(0, min(y, base_img.shape[0] - h))
        x = max(0, min(x, base_img.shape[1] - w))
        h = min(h, base_img.shape[0] - y)
        w = min(w, base_img.shape[1] - x)
        
        # Cut out the relevant parts of the base image and object
        cut_base = base_img[y:y+h, x:x+w]
        obj_img = obj_img[:h, :w]
        alpha_colored = alpha_colored[:h, :w]
        
        # Blend the object and background using the feathering mask
        blended = cv2.multiply(alpha_colored.astype(np.float32), obj_img.astype(np.float32)) + cv2.multiply(1.0 - alpha_colored.astype(np.float32), cut_base.astype(np.float32))
        
        # Replace the relevant part of the base image with the blended result
        base_img[y:y+h, x:x+w] = blended
        
        # Ensure the result is in the range [0, 255]
        base_img = np.clip(base_img, 0, 255).astype('uint8')
        
        return base_img


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
        occupancy_mask = labels.get('occupancy_mask', np.zeros_like(base_img[:,:, 0], dtype=np.uint8))

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

        # Position to insert object
        pos = (y2, x2)

        # Add noise to the object and insert it into the base image with feathered edges
        base_img[y2:y2+h2, x2:x2+w2] = self.feather_insert_object(
            base_img[y2:y2+h2, x2:x2+w2], 
            obj,
            binary_mask, 
            pos
        )

        # Get class id
        obj_class_id = suppl_labels['cls']

        # Add the new bounding box to the labels
        if 'cls' not in labels:
            # Create new tensors
            labels['cls'] = torch.tensor([[obj_class_id]])
            labels['batch_idx'] = torch.tensor([index])
            labels['masks'] = [segmentation_mask]
        else:
            # Append to labels
            labels['cls'] = torch.cat((labels['cls'], torch.tensor([[obj_class_id]])))
            labels['batch_idx'] = torch.cat((labels['batch_idx'], torch.tensor([index])))
            labels['masks'].append(segmentation_mask.copy())

        # Cast image back to tensor
        labels['img'] = self.convert_cv_to_tensor(base_img)

        # Store occupancy_mask
        labels['occupancy_mask'] = occupancy_mask

        return labels

    def filter_degenerate_masks(self, transformed: dict, min_visible_pixels: int = 4096, aspect_ratio_threshold: int = 4):
        class_labels = []
        masks = []
        for class_label, mask in zip(transformed['class_labels'], transformed['masks']):
            # Filter out objects that are too small
            if np.count_nonzero(mask) < min_visible_pixels:
                continue

            # Filter out objects with poor aspect ratios
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = contours[0]
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            try:
                if max(width, height) / min(width, height) > aspect_ratio_threshold:
                    continue
            except ZeroDivisionError:
                continue
            class_labels.append(torch.tensor(class_label))
            masks.append(torch.tensor(mask))
        return class_labels, masks

    def augment(self, labels: dict, augmentations: dict):
        transform_dict = defaultdict(list)
        for augmentation, chance in augmentations.items():
            if augmentation == 'random_crop':
                transform_dict['geometric'].append(
                        A.RandomCrop(
                            p=chance or 0.4,
                            height=random.randint(128, 368),
                            width=random.randint(128, 368),
                            )
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
                transform_dict['color'].append(
                        A.ChannelShuffle(p=chance or 0.4)
                        )
            if augmentation == 'color_jitter':
                transform_dict['color'].append(
                        A.ColorJitter(
                            p=chance or 0.4,
                            contrast=0.8,
                            brightness=0.8,
                            saturation=0.8,
                            hue=0.8,
                            )
                        )
            if augmentation == 'hsv_shift':
                transform_dict['color'].append(
                        A.HueSaturationValue(
                            p=chance or 0.4,
                            hue_shift_limit=80,
                            sat_shift_limit=80,
                            val_shift_limit=80,
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
            if augmentation == 'scale':
                transform_dict['geometric'].append(
                        A.Affine(
                            p=chance or 0.8,
                            scale=(0.05, 1.5),
                            keep_ratio=True,
                            )
                        )
            if augmentation == 'downscale':
                transform_dict['geometric'].append(
                        A.Downscale(
                            p=chance or 0.8,
                            scale_min=0.1,
                            scale_max= 0.8,
                            )
                        )
            if augmentation == 'rotate':
                transform_dict['geometric'].append(
                        A.Affine(
                            p=chance or 0.5,
                            rotate=[-30, 30],    # rotate 0 - 30deg in either direction
                        )
                        )
            if augmentation == 'optical_distortion':
                transform_dict['geometric'].append(
                        A.OpticalDistortion(p=chance or 0.5)
                        )
            if augmentation == 'grayscale':
                transform_dict['photometric'].append( 
                        A.ToGray(p=chance or 0.2),
                        )

        transforms = []
        # Limit transforms for performance reasons
        for category in transform_dict.keys():
            if category != 'photometric' or category != 'color':
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

        transforms = A.Compose(transforms)

        # Convert image to cv format
        img = self.convert_tensor_to_cv(labels['img'])
        
        # Make sure that the masks are lists of numpy arrays
        masks = labels.get('masks')
        if isinstance(masks, torch.Tensor):
            masks = [self.convert_tensor_to_cv(mask) for mask in masks]

        # Apply augmentations
        try:
            transformed = transforms(
                    image=img,
                    class_labels=labels.get('cls', []),
                    masks=masks,
                    )
        except Exception as e:
            print(f'{img=}')
            print(f'{labels.get("cls")=}')
            print(f'{masks=}')
            raise e

        # Convert image back to tensor
        labels['img'] = self.convert_cv_to_tensor(transformed['image'])

        if 'masks' not in labels:
            return labels

        class_labels, masks = self.filter_degenerate_masks(transformed)
        try:
            labels['cls'] = torch.stack(class_labels)
            labels['masks'] = torch.stack(masks)
            # Strip away batch indices for classes that got lost during augmentation
            labels['batch_idx'] = labels['batch_idx'][:len(class_labels)]
        except Exception as e:
            # Fill with placeholders if we can't stack
            labels['cls'] = torch.tensor([[-1]])
            labels['batch_idx'] = torch.tensor([-1])
            # pop mask key, this will keep the small masks 
            # from getting added to the image
            labels.pop('masks')
        return labels

    def add_placeholders(self, labels):
        if len(labels.get('batch_idx', [])) == 0:
            labels['batch_idx'] = torch.tensor([-1])

        if len(labels.get('cls', [])) == 0:
            labels['cls'] = torch.tensor([[-1]])

        for key in ['masks', 'occupancy_mask']:
            if len(labels.get(key, [])) == 0:
                labels[key] = torch.tensor(np.array([np.zeros_like(labels['img'][0, :, :])]))
        
        return labels
    
    def _get_objects(self):
        # Collect dataset into list that can be sampled
        dataset_list = list(self.suppl_dataset)
        while True:
            yield random.sample(dataset_list, k=int(random.uniform(1, self.max_pasted_objects)))


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

        # Create intermediate image to paste objects onto
        # They will then be augmented together, before getting pasted onto the background image
        intermediate_labels = copy.deepcopy(labels)
        intermediate_labels['img'] = np.zeros_like(labels['img'])

        # Choose randomly from the supplementary dataset
        objects_to_cut_list = next(self.object_sampler)
        # Paste individual objects onto intermediate image
        for obj_labels in objects_to_cut_list:
            intermediate_labels = self.copy_paste(intermediate_labels, obj_labels, self.batch_idx)

        # Augment intermediate image
        if self.augmentations:
            intermediate_labels = self.augment(intermediate_labels, augmentations=self.augmentations['object_level'])

        if 'masks' in intermediate_labels:
            # Transfer results onto original background image
            intermediate_img = labels['img']
            for mask in intermediate_labels['masks']:
                intermediate_img = self.feather_insert_object(
                    base_img=intermediate_img,
                    obj_img=intermediate_labels['img'],
                    binary_mask=mask,
                    pos=(0, 0),
                )

            labels = intermediate_labels
            labels['img'] = self.convert_cv_to_tensor(intermediate_img)
    
        # Augment entire image
        if self.augmentations and 'masks' in labels:
            labels = self.augment(labels, augmentations=self.augmentations['image_level'])

        labels = self.add_placeholders(labels)

        # Recompute bounding boxes for closer fit and assign to labels
        labels = self.recompute_bboxes(labels)

        self.batch_idx += 1
        self.batch_idx %= self.batch_size

        self._vis(labels, debug_view=False)
        return labels

    def _vis_batch(self, batch):
        print(f'{batch=}')
        for img, bbox in zip(batch['img'], batch['bboxes']):
            img = img.numpy().transpose((1, 2, 0)).copy()
            bboxes = bbox.numpy()
            print(f'{bboxes=}')
            for bbox in bboxes:
                center_x, center_y, width, height = bbox

                # Convert YOLO bounding box coordinates to OpenCV format
                if center_x < 1.:
                    center_x *= img.shape[1]
                    center_y *= img.shape[0]
                    width *= img.shape[1]
                    height *= img.shape[0]
                    print(f'{center_x=}, {width=}')
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                # Draw the bounding box
                print(f'{img.shape=}')
                cv2.rectangle(img, (x, y), (x + int(width), y + int(height)), (0, 255, 0), 2)

            # Draw the class label
            cv2.imshow('img', img)
            cv2.waitKey()

    def collate_fn(self, batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[[b[k] for k in keys] for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == 'img':
                value = torch.stack(value, 0)
            if k in ['keypoints', 'bboxes', 'cls']:
                try:
                    value = torch.cat(value, 0)
                except Exception as e:
                    print(f'{k=}: {value=}')
                    raise e
            new_batch[k] = value
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
        return new_batch

