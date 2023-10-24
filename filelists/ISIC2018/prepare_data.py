import os
import shutil
import pandas as pd

# Read the CSV file
df = pd.read_csv('/content/Dr_MAML/filelists/ISIC2018/ISIC2018_Task3_Training_GroundTruth.csv')

# Create the 'ISIC2018' directory
os.makedirs('/content/ISIC2018', exist_ok=True)

# Create subdirectories for each class
classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
for class_name in classes:
    os.makedirs(os.path.join('/content/ISIC2018', class_name), exist_ok=True)

# Copy the images to their respective class directories
for index, row in df.iterrows():
    image_name = row['image']
    image_class = row[classes].astype(float).idxmax()
    source_path = os.path.join('/content/ISIC2018_Task3_Training_Input', f'{image_name}.jpg')
    destination_path = os.path.join('ISIC2018',image_class, f'{image_name}.jpg')
    shutil.copy(source_path, destination_path)
