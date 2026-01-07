# SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import os

train_size = 0.7
valid_size = 0.1
# folders_path = "/fastwork/sberti/Diving48/images"
# annotations_path = "/fastwork/sberti/fsosar/splits/diving"
folders_path = "/fastwork/sberti/NTU_dataset/images"
annotations_path = "/fastwork/sberti/fsosar/splits/nturgbd"

# list folders in fodlers_path
folders = os.listdir(folders_path)

train_classes, validation_classes, test_classes = folders[:int(len(folders)*train_size)], folders[int(len(folders)*train_size):int(len(folders)*(train_size+valid_size))], folders[int(len(folders)*(train_size+valid_size)):]

print(f"Training classes: {len(train_classes)}, Validation classes: {len(validation_classes)}, Testing classes: {len(test_classes)}")
input()

# Open trainlist07.txt in annotations path
with open(os.path.join(annotations_path, "trainlist07.txt"), "w") as f:
    for folder in train_classes:
        for file in os.listdir(os.path.join(folders_path, folder)):
            f.write(folder + "/" + file + "\n")

with open(os.path.join(annotations_path, "vallist07.txt"), "w") as f:
    for folder in validation_classes:
        for file in os.listdir(os.path.join(folders_path, folder)):
            f.write(folder + "/" + file + "\n")

with open(os.path.join(annotations_path, "testlist07.txt"), "w") as f:
    for folder in test_classes:
        for file in os.listdir(os.path.join(folders_path, folder)):
            f.write(folder + "/" + file + "\n")

