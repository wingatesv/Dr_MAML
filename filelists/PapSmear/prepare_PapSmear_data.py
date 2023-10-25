import os

# Define the base directory
base_dir = '/content/smear2005/New database pictures'

# Walk through all subdirectories
for root, _, files in os.walk(base_dir):
    for filename in files:
        if filename.endswith('-d.bmp'):
            file_path = os.path.join(root, filename)
            os.remove(file_path)
            
print("Done PapSmear preparation!")
