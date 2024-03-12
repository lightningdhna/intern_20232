import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.insert(0, parent_dir)

import config

import sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

def build_rft_model():
    return RandomForestClassifier()    

def train_model(model,images,labels):
    images = images/255
    num_samples, height, width, num_channels = images.shape
    images = images.reshape((num_samples, height * width * num_channels))
    model.fit(images, labels)
    
def test_image(model,image_path):
    import cv2
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image2 = cv2.resize(image,config.image_wh)
    image2 = image2.reshape(1,config.image_pixel*3)
    y_hat = model.predict(image2)
    data_loader.show_image(image,y_hat)

def test_model(model, images, labels):
    # Flatten the images into 2D if they are not already
    if len(images.shape) > 2:
        num_images, height, width, num_channels = images.shape
        images = images.reshape((num_images, height * width * num_channels))

    # Predict labels for the images
    predicted_labels = model.predict(images)

    # Print the accuracy score
    print(accuracy_score(labels, predicted_labels))
    print(classification_report(labels, predicted_labels))

if __name__=="__main__":
    
    from modules import data_loader
    dataset = data_loader.tensorflow_load_data('data1/train')
    images, labels = data_loader.dataset_to_array(dataset)
    
    model = build_rft_model()
    train_model(model,images,labels)
    
    test_image(model,'models/dog.png')
    
    test_dataset = data_loader.tensorflow_load_data('data1/test')
    images, labels = data_loader.dataset_to_array(test_dataset)
    test_model(model,images,labels)
    
    print('finish')
