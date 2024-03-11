import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_preprocessing_config = config['data_preprocessing']


data_folder = data_preprocessing_config['data_folder']
image_size = data_preprocessing_config['image_size']
image_height =image_size[0]
image_width = image_size[1]
image_hw = (image_height,image_width)
image_wh = (image_width,image_height)
batch_size = data_preprocessing_config['batch_size']


dataset_config = config['dataset']

train_ratio = dataset_config['train_ratio']
val_ratio = dataset_config['val_ratio']
use_test_split = dataset_config['use_test_split']
test_ratio = 0
if use_test_split:
    test_ratio = dataset_config['test_ratio']   

saved_config = config['saved']

data_set_folder = saved_config['dataset_folder']
model_folder = saved_config['model_folder']
default_dataset_name = saved_config['default_dataset_name']
default_model_name = saved_config['default_model_name']


if __name__=="__main__":
    print(data_folder)
    print(image_size)
    print(image_height)
    print(image_width)
    print(batch_size)
    print(train_ratio)
    print(val_ratio)
    print(use_test_split)
    print(test_ratio)
    print(data_set_folder)
    print(model_folder)
    print(default_dataset_name)
    print(default_model_name)
    print(image_wh)
    print(image_hw)
    pass