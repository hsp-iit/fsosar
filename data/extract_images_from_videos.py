# SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import os
import cv2
import math
import tqdm

def extract_frames(input_dir):
    # Define paths
    videos_dir = os.path.join(input_dir, 'videos')
    images_dir = os.path.join(input_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Iterate over folders inside 'videos' directory
    for class_folder in tqdm.tqdm(os.listdir(videos_dir)):
        class_path = os.path.join(videos_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        # Create corresponding class folder in 'images' directory
        class_images_path = os.path.join(images_dir, class_folder)
        os.makedirs(class_images_path, exist_ok=True)

        # Process each video inside the class folder
        for video_file in tqdm.tqdm(os.listdir(class_path)):
            video_path = os.path.join(class_path, video_file)
            if not os.path.isfile(video_path):
                continue

            # To continue script execution, if output folder exists, check integrity
            video_file = os.path.splitext(video_file)[0]
            images_path = os.path.join(class_images_path, video_file)
            if os.path.exists(images_path):
                if len(os.listdir(images_path)) == 16:
                    print(f"Skipping video {images_path}, frames already extracted.")
                    continue
                else:
                    print(f"Class {class_folder} has missing frames, re-extracting")
                    # Erase file in folder
                    for file in os.listdir(images_path):
                        os.remove(os.path.join(images_path, file))
            else:
                os.mkdir(images_path)

            # Read the video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error opening video file: {video_path}")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count < 16:
                print(f"Video {video_file} has less than 16 frames, skipping.")
                cap.release()
                continue

            # Calculate frame indices to capture
            frame_indices = [math.floor(i * frame_count / 16) for i in range(16)]

            # Extract and save frames
            frame_counter = 0
            frames_extracted = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_counter in frame_indices:
                    frame_output_path = os.path.join(images_path, f"{frames_extracted}.jpg")

                    cv2.imwrite(frame_output_path, frame)
                    frames_extracted += 1

                frame_counter += 1

                if frames_extracted >= 16:
                    break

            cap.release()
            print(f"Processed video: {video_file}, frames saved to {class_images_path}")

if __name__ == "__main__":
    print("To make this script work, give the path to the dataset root directory, that must contain a 'videos' folder.")
    print("The video folder must contain foldlers named as classes with videos inside them.")
    print("For SSv2, move all videos inside a directory, such that they are all treated as a single class.")
    input_directory = "/home/steb6/datasets/SSv2"  # Replace with your input directory path
    extract_frames(input_directory)
