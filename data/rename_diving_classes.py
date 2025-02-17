import os


# i want to give unique names to folders inside Diving, otherwise it doesnt work
diving_path = "/home/sberti_datasets/Diving48/images"

for folder in os.listdir(diving_path):
    class_folders = os.path.join(diving_path, folder)
    for class_folder in os.listdir(class_folders):
        os.rename(os.path.join(class_folders, class_folder), os.path.join(class_folders, folder+"_"+class_folder))
