import cv2
import os
import hashlib
from PIL import Image, UnidentifiedImageError
import imagehash
import imghdr
import re
import yaml
import warnings

config = None

with open('config.yaml','r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)['data_preprocessing']

if(config is not None):
    data_folder = config.get('data_folder')

def check_and_get_index(filename, category):
    pattern = re.compile(rf'^{category}_(\d+)\.jpeg$')  
    match = pattern.match(filename)
    if match:
        return int(match.group(1))
    else:
        return None
    
def is_valid_image(file_path):
    # Check if the file is an image
    if imghdr.what(file_path) is None:
        return False

    # Check if the image can be opened
    try:
        with warnings.catch_warnings():    
            warnings.simplefilter("ignore")
            with Image.open(file_path) as img:
                if img.mode == 'P':  # check if image is a palette type
                    img = img.convert('RGBA')
                img.verify()
    except (IOError, SyntaxError, UnidentifiedImageError):
        return False

    return True

    
def remove_invalid_images(directory = None, categories =None):
    if directory == None:
        directory = data_folder

    if not categories:
        categories = os.listdir(directory)

    for category in categories:
        
        path = os.path.join(directory, category)
        # print(path)
        if not os.path.isdir(path):
            continue
        for f in os.listdir(path):
            file_path = os.path.join(path, f)
            if not is_valid_image(file_path):
                os.remove(file_path)
                
                


def rename_images(directory, categories =None):
    if not categories:
        categories = os.listdir(directory)

    for category in categories:
        
        path = os.path.join(directory, category)
        if not os.path.isdir(path):
            continue
        
        
        image_list = []

        # rename first time
        for f in os.listdir(path):
            if imghdr.what(os.path.join(path, f)) is not None:
                old_name = os.path.join(path, f)
                base_name, ext = os.path.splitext(old_name)
                new_name = f"{base_name}fff{ext}"
                os.rename(old_name, new_name)
                image_list.append(new_name)
                
                
        #rename second time
        index = int(0)
        base_name = f"{path}/{category}_{{}}.jpeg"
        
        for file_name in image_list:
            os.rename(file_name,base_name.format(index))
            index += 1
        
        
                
                
                    
from IPython.display import display

def remove_duplicate_images(directory, categories=None):
    def get_image_hash(file_path):
        with Image.open(file_path) as img:
            return str(imagehash.phash(img))

    if not categories:
        categories = os.listdir(directory)

    for category in categories:
        path = os.path.join(directory, category)
        if not os.path.isdir(path):
            continue

        image_hashes = {}
        for f in os.listdir(path):
            file_path = os.path.join(path, f)
            if imghdr.what(file_path) is not None:
                image_hash = get_image_hash(file_path)
                if image_hash in image_hashes:
                    # Display the image and its name before deleting it

                    # display(Image.open(file_path))
                    # img = cv2.imread(file_path)
                    # cv2.imshow('Duplicate Image', img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    
                    print(f'Removing duplicate image: {file_path}')
                    os.remove(file_path)
                else:
                    image_hashes[image_hash] = file_path
