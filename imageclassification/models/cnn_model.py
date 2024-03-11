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
    model.add(Flatten())
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    
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
    
def test_model(model,test_ds):
    from keras.metrics import Precision, Recall, BinaryAccuracy
    pre = Precision()
    re = Recall()
    # acc = BinaryAccuracy()
    for batch in test_ds.as_numpy_iterator(): 
        X, y = batch
        y_hats = model.predict(X)
        pre.update_state(y, y_hats)
        re.update_state(y, y_hats)
        # acc.update_state(y, y_hats)
    print(f"precision : {pre.result().numpy().squeeze()}")
    print(f"recall : {re.result().numpy().squeeze()}")
    
def show_train_loss(history):
    import matplotlib.pyplot as plt
    plt.plot(history.history['loss'], color='red', label='loss')
    plt.plot(history.history['val_loss'], color='teal', label='val_loss')
    
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()
    
def show_train_accuracy(history):
    import matplotlib.pyplot as plt
    
    plt.plot(history.history['accuracy'], color='teal', label='accuracy')
    plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
    
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    
    plt.legend(loc="lower right")
    
    plt.show()
    

    
def image_classification():
    from modules import data_loader
    
    model = build_simple_cnn_model()
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    # data = data_loader.load_saved_dataset()
    data = data_loader.tensorflow_load_data('data1/train')
    data = data.map(lambda x,y:(x/255.0,y))
    batch = data.as_numpy_iterator().next()
    data_loader.show_image(batch[0],batch[1])
    train_ds,val_ds,test_ds = data_loader.split_data(data)

    hist = train_model(model,train_ds=train_ds,val_ds=val_ds,epochs=50)
    show_train_accuracy(hist)
    save_model(model)
    test_model(model,test_ds)
    
    test_model(model,test_ds=test_ds)
    
def ic_test():
    model = load_saved_model()
    test_image(model,'models/cat.png')
    test_image(model,'models/dog.png')
    
if __name__=="__main__":
    # image_classification()
    ic_test()
    pass

    
    