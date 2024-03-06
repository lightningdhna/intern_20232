

import math
import cv2
import yaml
import os
from matplotlib import pyplot as plt
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)

config_file = os.path.join(script_dir,'..', 'config.yaml')
with open (config_file) as f:
    config = yaml.safe_load(f)['data_preprocessing']
data_folder = config['data_folder']
data_folder_path = os.path.join(script_dir,'..' ,data_folder)
print(data_folder_path)

categories = os.listdir(data_folder_path)
image_size = config['image_size']
image_size = [int(i) for i in image_size]
image_width = image_size[0]
image_height = image_size[1]
print(image_size)
print(f"Image size: {image_width} x {image_height}")

batch_size = config['batch_size']
    
    
def show_image(images,labels = None):

    if isinstance(images,list) or isinstance(images,np.ndarray) and images.ndim==4:
        if labels is None:
            labels =[]

        images_len = int(len(images))
        labels_len = int(len(labels))
        
        if labels_len < images_len:
            labels.extend([""]*(images_len - labels_len))
        
        grid_size  = math.isqrt(images_len)

        
         # Create subplots
        fig, axs = plt.subplots(grid_size, grid_size+1)
        # Flatten the array of axes
        axs = axs.ravel()

        # Show all images
        for i,(image,label) in enumerate(zip(images,labels)):
            
            axs[i].imshow(image)
            axs[i].set_title(f'{label}', color='blue')
            axs[i].axis('off')

        # Remove unused subplots
        for i in range(images_len,grid_size*(grid_size+1)):
            fig.delaxes(axs[i])
    else:
        plt.imshow(images)
        if not isinstance(labels,list):
            if labels is not None:
                plt.text(0.5, 0.5, labels, color='red', ha='center', va='center', transform=plt.gca().transAxes)
    plt.show()
    
def load_data(directory = None ,categories = None):
    images = []
    labels = []
    if directory is None:
        directory = data_folder_path
    if categories is None:
        categories = os.listdir(directory)
        
    for category in categories:
        path = os.path.join(directory,category)
        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            image = cv2.imread(img_path)
            # show_image(image)
            image = cv2.resize(image,(image_width,image_height))
            image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            images.append(image)
            labels.append(category)
             
            print(img_path)
    return images,labels
def tensorflow_load_data(directory=None):
    import tensorflow as tf
    
    if directory is None:
        directory = data_folder_path
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(image_height, image_width),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False)
    pass
