import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import shutil
from torchvision import transforms as T
from PIL import Image

emotions = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}

# input: pcsv file path
# output: The two directory paths contain training data and testing data

class FaceExpressionDataset():

    def __init__(self, path_data_csv:str):
        super().__init__()
        self.path_data_csv = path_data_csv
        self.data = pd.read_csv(path_data_csv)


    def create_dirs(self, relative_dir):
        if os.path.exists(f'{relative_dir}/train'):
            shutil.rmtree(f'{relative_dir}/train')
            print('removed dir train')
        if os.path.exists(f'{relative_dir}/test'):
            shutil.rmtree(f'{relative_dir}/test')
            print('removed dir test')


        for emotion in emotions.values():
            os.makedirs(f'{relative_dir}/train/{emotion}', exist_ok=True)
            os.makedirs(f'{relative_dir}/test/{emotion}', exist_ok=True)
        print('created directories')

    def save_image(self, image_arr, file_path:str):
        img = Image.fromarray(image_arr)
        img.save(file_path)

    
    def pre_data(self):
        relative_dir = os.path.relpath(os.path.dirname(os.path.abspath(__file__)))
        self.create_dirs(relative_dir)
        for i,row in tqdm(self.data.iterrows(), total=self.data.shape[0], desc='Processing'):
        # for i,row in self.data.iterrows():

            emotion = row['emotion']
            pixels =  row['pixels']
            usage = row['Usage']

            image_arr = np.fromstring(pixels,sep=' ').reshape(48,48)
            image_arr = np.uint8(image_arr)

            #save image 

            if usage == 'Training':
                self.save_image(image_arr,f'{relative_dir}/train/{emotions[int(emotion)]}/{i}.png')

            if usage == 'PublicTest' or usage == 'PrivateTest':
                self.save_image(image_arr,f'{relative_dir}/test/{emotions[int(emotion)]}//{i}.png')
        print('Data preprocessing is done')
    
        return f'{relative_dir}/train', f'{relative_dir}/test'


if __name__ == "__main__":

    dataset = FaceExpressionDataset(r"data\fer2013.csv")
    path_1, path_2= dataset.pre_data()
    print(path_1)
    print(path_2)
    # FaceExpressionDataset(r"data\fer2013.csv")
# relative_dir = os.path.relpath(os.path.dirname(os.path.abspath(__file__)))
# print(f'{relative_dir}/train')
# print( f'{relative_dir}/train')





    