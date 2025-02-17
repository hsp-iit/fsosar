import os

train_size = 0.7
folders_path = "/home/sberti_datasets/Diving48/images"
annotations_path = "/home/iit.local/sberti/projects/fsosar/splits/diving"

# list folders in fodlers_path
folders = os.listdir(folders_path)

train_classes, test_classes = folders[:int(len(folders)*train_size)], folders[int(len(folders)*train_size):]

# Open trainlist07.txt in annotations path
with open(os.path.join(annotations_path, "trainlist07.txt"), "w") as f:
    for folder in train_classes:
        for file in os.listdir(os.path.join(folders_path, folder)):
            f.write(folder + "/" + file + "\n")

with open(os.path.join(annotations_path, "testlist07.txt"), "w") as f:
    for folder in test_classes:
        for file in os.listdir(os.path.join(folders_path, folder)):
            f.write(folder + "/" + file + "\n")

