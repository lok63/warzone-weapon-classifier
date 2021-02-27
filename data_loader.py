from PIL import Image
import pathlib
from matplotlib import pyplot as plt
from numpy import asarray
import numpy as np
from typing import Tuple
import os
import pandas as pd

class PLIWrapper():

    @staticmethod
    def image_print_meta(image : Image) -> None:
        print(image.format)
        print(image.mode)
        print(image.size)

    @staticmethod
    def plot_image(img) -> None:
        plt.imshow(img)
        plt.show()

    @staticmethod
    def img_to_array(img: Image) -> np.ndarray:
        return asarray(img)

    @staticmethod
    def array_to_img(array : np.ndarray) -> Image:
        return Image.fromarray(array)

    @staticmethod
    def to_grayscale(img : Image) -> Image:
        return img.convert(mode='L')

    @staticmethod
    def resize(img : Image, dim : Tuple) -> Image :
        return img.resize(dim)

    @staticmethod
    def flip(img : Image, vertical : bool = False) -> Image:
        return img.transpose(Image.FLIP_TOP_BOTTOM) if vertical else img.transpose(Image.FLIP_LEFT_RIGHT)

    @staticmethod
    def save(img : Image, path : str) -> None :
        img.save(path)


if __name__ == '__main__':
    pli = PLIWrapper()

    def process_image(path):
        img = Image.open(pathlib.Path.cwd().joinpath('datasets', 'weapon', 'test', 'amax', '1.PNG'))
        grayscaled = pli.to_grayscale(img)
        resized = pli.resize(grayscaled, (120, 40))

        data = pli.img_to_array(resized)
        data1d = data.flatten()
        return data1d

    labels = {'amax': 0,
              'ffar' : 1,
              'grau' : 2,
              'kilo' : 3,
              'mac10' : 4}

    train_images = []
    test_images = []

    root_train = pathlib.Path.cwd().joinpath('datasets', 'weapon', 'train')
    root_test = pathlib.Path.cwd().joinpath('datasets', 'weapon', 'test')



    for l in os.listdir(root_test):
        for pic in os.listdir(root_train.joinpath(l)):
            full_path = root_train.joinpath(l, pic)
            img_data = process_image(full_path)
            train_images.append({'label':labels[l], 'pixels':img_data})

    for l in os.listdir(root_test):
        for pic in os.listdir(root_test.joinpath(l)):
            full_path = root_test.joinpath(l, pic)
            img_data = process_image(full_path)
            test_images.append({'label':labels[l], 'pixels':img_data})


    # Save data to csv files
    df_train = pd.DataFrame(train_images, columns=['label', 'pixels'])
    df_test = pd.DataFrame(test_images, columns=['label', 'pixels'])

    df_train.to_csv(pathlib.Path.cwd().joinpath('datasets', 'csv', 'train.csv'), header=True, index=False)
    df_test.to_csv(pathlib.Path.cwd().joinpath('datasets', 'csv', 'test.csv'),  header=True, index=False)










