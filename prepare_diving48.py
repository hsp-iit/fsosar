import json
import os
import numpy as np
import cv2

def extract_equidistant_frames(video_path, num_frames=16, output_folder="extracted_frames"):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval between frames
    interval = total_frames // num_frames

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through the video and extract frames
    for i in range(num_frames):
        # Set the frame position
        frame_id = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        # Read the frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {frame_id}")
            break

        # Save the frame
        frame_filename = os.path.join(output_folder, f"{i}.jpg")
        cv2.imwrite(frame_filename, frame)
        # print(f"Saved {frame_filename}")

    # Release the video capture object
    cap.release()
    # print("Extraction complete.")

def load_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    train_split = "/home/sberti_datasets/Diving48/Diving48_V2_train.json"
    test_split = "/home/sberti_datasets/Diving48/Diving48_V2_test.json"
    data_path = "/home/sberti_datasets/Diving48/rgb"
    dataset_path = "/home/sberti_datasets/Diving48"
    class_labels = "/home/sberti_datasets/Diving48/class_labels.json"

    train_data = load_json(train_split)
    test_data = load_json(test_split)
    class_labels = load_json(class_labels)

    id_to_class = {}
    for x in train_data:
        id_to_class[x['vid_name']] = x['label']
    for x in test_data:
        id_to_class[x['vid_name']] = x['label']

    id_to_label = []
    for c in class_labels:
        id_to_label.append("_".join(c))
    
    total = len(id_to_class)

    # add tqdm progress bar
    print(f"Extracting frames for {total} videos")
    count = 0

    # iterate all videos inside data_path
    for root, dirs, files in os.walk(data_path):
        for file in files:
            print(f"\r{count}/{total}", end="")
            count += 1
            if file.endswith(".mp4"):
                vid_name = file.split(".")[0]
                try:
                    label = id_to_class[vid_name]
                except KeyError:
                    print(f"Error: Could not find label for {vid_name}")
                    continue

                output_path = os.path.join(dataset_path, "images", id_to_label[label])
                
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                # get number of files in the directory to assign lowest id to this video
                num_files = len([name for name in os.listdir(output_path)])

                output_path = os.path.join(output_path, str(num_files))
                os.makedirs(output_path)

                # extract frames
                extract_equidistant_frames(os.path.join(data_path, file), output_folder=output_path)
                

    print("")