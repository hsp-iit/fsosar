import json
import os
import shutil
import tqdm

splits_path = "/home/steb6/projects/fsosar/splits/ssv2_OTAM"
train_split_path = os.path.join(splits_path, "trainlist07.txt")
val_split_path = os.path.join(splits_path, "vallist07.txt")
test_split_path = os.path.join(splits_path, "testlist07.txt")

input_path = "/home/steb6/datasets/SSv2/images"
output_path = "/home/steb6/datasets/SSv2/images_in_class_folders"

# Open split_id -> classnames
with open(os.path.join(splits_path, "/home/steb6/projects/fsosar/ssv2_from_splitid_to_classname.json"), "r") as f:
    split_id_to_classname = json.load(f)

# Open splits
for split in [train_split_path, val_split_path, test_split_path]:
    with open(split, "r") as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            line = line.strip()
            split_id, video_id = line.split("/")
            class_folder = split_id_to_classname[split_id]
            class_folder = class_folder.lower().replace(" ", "_").replace('[' , '').replace(']', '')
            output_folder = os.path.join(output_path, class_folder)

            # Test after launching the script first time
            assert os.path.exists(output_folder)

            # Move video to class
            # if not os.path.exists(output_folder):
            #     os.mkdir(output_folder)
            # video_path = os.path.join(input_path, video_id)
            # shutil.move(video_path, output_folder)
            # print(f"Moved {video_path} to {output_folder}")