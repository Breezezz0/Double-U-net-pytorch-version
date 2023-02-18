import torch
import torch.nn as nn
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from unet.unet_model import UNet , Double_unet
import numpy as np
from tqdm import tqdm
from dataset import EndoscopyDataset, visualize,colour_code_segmentation,reverse_one_hot,get_preprocessing
from loss import DiceLoss
import torch.optim as optim
import segmentation_models_pytorch as smp


def run():
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
    print('""bulid dataset""')
    preprocessing_fn = smp.encoders.get_preprocessing_fn('vgg19', pretrained='imagenet')
    train_dataset = EndoscopyDataset(
        train_df,
        class_rgb_values=select_class_rgb_values,
        preprocessing= get_preprocessing(preprocessing_fn)
    )
    image, mask = train_dataset[0]
    valid_dataset = EndoscopyDataset(
        valid_df,
        class_rgb_values=select_class_rgb_values,
        preprocessing= get_preprocessing(preprocessing_fn)
    )
    test_dataset = EndoscopyDataset(
        test_df,
        class_rgb_values=select_class_rgb_values,
        preprocessing= get_preprocessing(preprocessing_fn)
    )
    test_dataset_vis = EndoscopyDataset(
        test_df,
        class_rgb_values=select_class_rgb_values
)
    print(train_dataset[0][0].shape)
    # Get train and val data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=0)
    valid_loader = DataLoader(
        valid_dataset, batch_size=8, shuffle=False, num_workers=0)
    print('""""end build dataset""""')
    model = Double_unet(n_channels=image.shape[0], n_classes=2)
    model.load_state_dict(torch.load('model_test4.pth'))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    dsc_loss = DiceLoss()
    epochs = 50
    loss_train = []
    loss_valid = []
    print('start training')
    for i in tqdm(range(epochs), total=epochs):
        for phase in ['train', 'validation'] :
            if phase == 'train ' :
                model.train()
                for batch in train_loader:
                    x, y_true = batch
                    x, y_true = x.float().to(DEVICE), y_true.to(DEVICE)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        y_pred = model(x)

                        loss = dsc_loss(y_pred, y_true)
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()
            else :
                model.eval()
                validation_pred = []
                validation_true = []
                for batch in valid_loader:
                    x, y_true = batch
                    x, y_true = x.float().to(DEVICE), y_true.to(DEVICE)
                    with torch.set_grad_enabled(mode = False):
                        y_pred = model(x)

                        loss = dsc_loss(y_pred, y_true)
                        loss_valid.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend(
                            [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                        )
                        y_true_np = y_true.detach().cpu().numpy()
                        validation_true.extend(
                            [y_true_np[s] for s in range(y_true_np.shape[0])]
                        )

    print('end training')
    torch.save(model.state_dict(), r'./model_test6.pth')
    print(image.shape)

    print('"""start testing"""')
    #model.load_state_dict(torch.load('model_test5.pth'))
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=0)
    loss_test = []
    model.eval()
    for batch in test_loader:
        x, y_true = batch
        x, y_true = x.float().to(DEVICE), y_true.to(DEVICE)
        with torch.set_grad_enabled(mode = False):
            y_pred = model(x)

            loss = dsc_loss(y_pred, y_true)
            loss_test.append(loss)
    print('""visualize one sample""')
    image , mask = test_dataset[0]
    image_vis  = test_dataset_vis[0][0].astype('uint8')
    print(image.shape)
    image_input = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    print(mask[:,1,1])
    #print(image_input)
    #print(image)
    y_pred = model(image_input)
    y_pred_onehot = y_pred.detach().cpu().numpy()
    y_softmax = F.softmax(y_pred).detach().cpu().numpy()
    print('y_softmax=\n', y_softmax)
    print(y_pred_onehot[0,:,1,1])
    # y_pred_class1_1 = y_pred_onehot[0,0,:,:]
    # y_pred_class1_2 = y_pred_onehot[0,1,:,:]
    # y_pred_class1 = ((y_pred_class1_1 + y_pred_class1_2)/2) >0.5
    # y_pred_class2 = ((y_pred_class1_1 + y_pred_class1_2)/2) <=0.5
    # y_pred_class1 = np.expand_dims(y_pred_class1 , axis=0)
    # y_pred_class2 = np.expand_dims(y_pred_class2 , axis=0)
    # y_class_map = np.concatenate([y_pred_class1,y_pred_class2],axis=0)
    # y_class_map = np.expand_dims(y_class_map , axis=0 )
    # print(y_class_map.shape)

    y_pred_onehot = np.where(y_softmax > 0.5,1,0 )
    y_pred_onehot = np.squeeze(y_pred_onehot)
    y_pred_onehot = np.transpose(y_pred_onehot,(1,2,0))
    mask = np.transpose(mask , (1,2,0))
    # print(y_pred_onehot)
    # print(mask)
    # print(image_vis)
    #mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values)
    # mask = reverse_one_hot(mask)
    # print(mask.shape)
    visualize(
        original_image = image_vis,
        ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
        predicted_mask = colour_code_segmentation(reverse_one_hot(y_pred_onehot), select_class_rgb_values)
    )
    

    #print(y_pred_rgb.shape)
    
      

        
    print('"""finish testing"""')


if __name__ == '__main__':
    run()
    