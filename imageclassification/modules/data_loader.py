
import sys
import os


# Get the directory containing the current file
current_directory = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_directory = os.path.dirname(current_directory)
print(parent_directory)
# Add the parent directory to sys.path
sys.path.insert(0, parent_directory)

import config
import math
import cv2
from matplotlib import pyplot as plt
import numpy as np

data_folder_path = os.path.join(parent_directory,config.data_folder)
categories = os.listdir(data_folder_path)
    
def show_image(images,labels = None):

    if isinstance(images,list) or isinstance(images,np.ndarray) and images.ndim==4:
        if labels is None:
            labels =[]

        images_len = int(len(images))
        labels_len = int(len(labels))
        
        if labels_len < images_len:
            labels.extend([""]*(images_len - labels_len))
        
        grid_size  = int(math.ceil(math.sqrt(images_len)))
        
        # Create subplots
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        fig.tight_layout()
        # Flatten the array of axes
        axs = axs.ravel()

        # Show all images
        for i,(image,label) in enumerate(zip(images,labels)):
            
            axs[i].imshow(image)
            axs[i].set_title(f'{label}', color='blue')
            axs[i].axis('off')

        # Remove unused subplots
        for i in range(images_len,grid_size*(grid_size)):
            fig.delaxes(axs[i])
    else:
        plt.imshow(images)
        if not isinstance(labels,list) and labels is not None:
            plt.text(0.5, 0.5, labels, color='red', ha='center', va='center', transform=plt.gca().transAxes)
    plt.show()
    
def load_data(directory = data_folder_path ,categories = None):
    images = []
    labels = []

    if categories is None:
        categories = os.listdir(directory)
        
    for category in categories:
        path = os.path.join(directory,category)
        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            image = cv2.imread(img_path)
            # show_image(image)
            image = cv2.resize(image,(config.image_width,config.image_height))
            image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            images.append(image)
            labels.append(category)
             
            print(img_path)
    return images,labels
def tensorflow_load_data(directory=data_folder_path):
    import tensorflow as tf
    if directory is None:
        return None
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=config.batch_size,
        image_size=(config.image_height, config.image_width),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False)


def save_dataset(dataset,\
            dataset_name = config.default_dataset_name,\
            dataset_folder = config.data_set_folder ):
    path = os.path.join(parent_directory,dataset_folder,dataset_name)
    import shutil
    shutil.rmtree(path)
    os.mkdir(path)
        
    import tensorflow as tf
    tf.data.Dataset.save(dataset,path=path)
    
def load_saved_dataset(dataset_name = config.default_dataset_name,\
            dataset_folder = config.data_set_folder ):
    import tensorflow as tf
    return tf.data.Dataset.load(os.path.join(parent_directory,dataset_folder,dataset_name))

def split_data(data):
    
    train_size = int(len(data)*config.train_ratio)
    val_size = int(len(data)*config.val_ratio)
    test_size = int(len(data)*config.test_ratio)
    
    train_ds = data.take(train_size)
    val_ds = data.skip(train_size).take(val_size)
    test_ds = data.skip(train_size+val_size).take(test_size)
    
    return train_ds,val_ds,test_ds

def dataset_to_array(dataset):
    images = []
    labels = []
    for batch in dataset:
        images.append(batch[0].numpy())
        labels.append(batch[1].numpy())
    images = np.concatenate(images,axis = 0)
    labels = np.concatenate(labels,axis = 0)
    return images,labels

if __name__=="__main__":
    print(1)
    print(categories)
    data = tensorflow_load_data()
    save_dataset(data)
    