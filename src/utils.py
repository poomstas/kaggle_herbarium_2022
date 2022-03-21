import os
import pandas as pd

def change_img_data_root_directory(csv_file_path, img_data_root):
    df = pd.read_csv(csv_file_path)
    df['directory'] = [os.path.join(img_data_root, '/'.join(subpath.split('/')[6:])) for subpath in df['directory']]
    df.to_csv(csv_file_path, index=False)

if __name__=='__main__':
    # Change image data root directory in csv files ( for both test and train)
    csv_file_path = '../test.csv'
    img_data_root = '/home/brian/dataset/herbarium_2022/'
    change_img_data_root_directory(csv_file_path, img_data_root)

    csv_file_path = '../train.csv'
    img_data_root = '/home/brian/dataset/herbarium_2022/'
    change_img_data_root_directory(csv_file_path, img_data_root)
