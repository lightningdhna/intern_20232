import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

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

def build_simple_cnn_model():
    from keras.models import Sequential
    from keras.layers import Conv2D,MaxPooling2D,Dense, Flatten,Dropout, Input
    model = Sequential()
    model.add(Input(shape = config.image_size))
    model.add(Conv2D(32, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(8, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(512, activation='tanh'))
    model.add(Dense(512, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.summary()
    return model

def train_model(model,train_ds,val_ds,epochs):
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    return history

def save_model(model,\
        model_name = config.default_model_name,\
        model_folder = config.model_folder):
    path = os.path.join(parent_directory,model_folder)
    if not os.path.exists(path):
        os.mkdir(path)
    model.save(os.path.join(path,model_name))
    
    pass

def load_saved_model(model_name = config.default_model_name,\
                    model_folder = config.model_folder):
    import keras
    path = os.path.join(parent_directory,model_folder,model_name)
    return keras.models.load_model(path)

def test_image(model,image_path):
    import cv2
    image = cv2.imread(image_path)
    image = cv2.resize(image,config.image_wh)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image/255
    
    import numpy as np
    y_hat = model.predict(np.expand_dims(image,axis = 0))
    
    y_hat = y_hat.squeeze()
    print(y_hat)
    
 # coordinates of the bottom-left corner of the text
    font = cv2.FONT_HERSHEY_SIMPLEX  # font type
    scale = 1  # font scale
    color = (255, 0, 0)  # color in BGR
    cv2.putText(image,str(y_hat),(50,50),fontScale=scale , fontFace=font, color=color)
    from modules import data_loader
    data_loader.show_image(image,str(y_hat))
    

if __name__=="__main__":
    model = build_simple_cnn_model()
    save_model(model=model)
    model = load_saved_model()
    model.summary()
    # test_image(model,'data/cat/cat_1.jpeg')
    print(1)