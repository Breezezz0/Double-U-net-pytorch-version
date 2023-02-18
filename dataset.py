import os
import cv2
import numpy as np
import pandas as pd
import random
import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album
import segmentation_models_pytorch as smp


def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

def draw_image(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    fig = plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)
    return fig


# Perform one hot encoding on label


def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map

# Perform reverse one-hot-encoding on labels / preds


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis=-1)
    return x

# Perform colour coding on the reverse-one-hot outputs


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)


class EndoscopyDataset(torch.utils.data.Dataset):

    """CVC-ClinicDB Endoscopic Colonoscopy Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        df (str): DataFrame containing images / labels paths
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            df,
            class_rgb_values=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.image_paths = df['png_image_path'].tolist()
        self.mask_paths = df['png_mask_path'].tolist()

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read images and masks
        image = cv2.cvtColor(cv2.imread(
            self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)

        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']


        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask

    def __len__(self):
        # return length of
        return len(self.image_paths)
class EndoscopyDataset_withonehot(torch.utils.data.Dataset):


    def __init__(
            self,
            df,
            class_rgb_values=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.image_paths = df['png_image_path'].tolist()
        self.mask_paths = df['png_mask_path'].tolist()

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read images and masks
        image = cv2.cvtColor(cv2.imread(
            self.image_paths[i]), cv2.COLOR_BGR2RGB)
        image = np.clip(image - np.median(image)+127, 0, 255)
        image = image/255.0
        image = image.astype(np.float32)
        image = np.transpose(image , (2,0,1))
        
        mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32)
        mask = mask/255.0
        mask = np.expand_dims(mask, axis=0)
        mask = np.concatenate([mask, mask],axis=0)

        
        return image, mask

    def __len__(self):
        # return length of
        return len(self.image_paths)


if __name__ == '__main__':
    DATA_dir = r'C:\Users\user\Downloads\CVC-ClinicDB'

    metadata_df = pd.read_csv(os.path.join(DATA_dir, 'metadata.csv'))
    metadata_df = metadata_df[['frame_id', 'png_image_path', 'png_mask_path']]
    metadata_df['png_image_path'] = metadata_df['png_image_path'].apply(
        lambda img_pth: os.path.join(DATA_dir, img_pth))
    metadata_df['png_mask_path'] = metadata_df['png_mask_path'].apply(
        lambda img_pth: os.path.join(DATA_dir, img_pth))
    # Shuffle DataFrame
    metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)

    # P erform 70/20/10 split for train / val / test
    valid_df = metadata_df.sample(frac=0.2, random_state=42)
    train_df = metadata_df.drop(valid_df.index)
    test_df = train_df.sample(n=int(0.5*len(valid_df.index)))
    train_df = train_df.drop(test_df.index)

    print(len(train_df), len(valid_df), len(test_df))
    class_dict = pd.read_csv(os.path.join(DATA_dir, 'class_dict.csv'))
    # Get class names
    class_names = class_dict['class_names'].tolist()
    # Get class RGB values
    class_rgb_values = class_dict[['r', 'g', 'b']].values.tolist()

    print('All dataset classes and their corresponding RGB values in labels:')
    print('Class Names: ', class_names)
    print('Class RGB values: ', class_rgb_values)
    # helper function for data visualization
    # Useful to shortlist specific classes in datasets with large number of classes
    select_classes = ['background', 'polyp']

    # Get RGB values of required classes
    select_class_indices = [class_names.index(
        cls.lower()) for cls in select_classes]
    select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

    print('Selected classes and their corresponding RGB values in labels:')
    print('Class Names: ', class_names)
    print('Class RGB values: ', class_rgb_values)

    dataset = EndoscopyDataset(
        train_df, class_rgb_values=select_class_rgb_values,preprocessing=get_preprocessing())
    random_idx = random.randint(0, len(dataset)-1)
    image, mask = dataset[2]
    print(image.shape)
    print(mask.shape)
    # visualize(
    #     original_image=image,
    #     ground_truth_mask=colour_code_segmentation(
    #         reverse_one_hot(mask), select_class_rgb_values),
    #     one_hot_encoded_mask=reverse_one_hot(mask)
    # )
    # image = cv2.cvtColor(cv2.imread(
    #     train_df['png_image_path'][0]), cv2.COLOR_BGR2RGB)
    # # image =
    # image = np.moveaxis(image, 2, 0)
    # print(image.shape)
    # mask = cv2.cvtColor(cv2.imread(
    #     train_df['png_mask_path'][0]), cv2.COLOR_BGR2RGB)
    # mask = one_hot_encode(mask, class_rgb_values).astype('float')
    # print(mask.shape)
