#!/usr/bin/env python3
import os
import time

# Set your target directory here
TARGET_DIR = "/path/to/target/directory"

def clean_directories(target_dir):
    # Iterate over each entry in the target directory
    for entry in os.scandir(target_dir):
        if entry.is_dir():
            dir_path = entry.path
            files = []
            # Gather all files in the current directory
            for file_entry in os.scandir(dir_path):
                if file_entry.is_file():
                    files.append(file_entry.path)
            # If there are files, determine the most recent one and delete the others
            if files:
                # The file with the maximum creation time
                newest_file = max(files, key=os.path.getctime)
                for file_path in files:
                    if file_path != newest_file:
                        try:
                            # os.remove(file_path)
                            print(f"Removed: {file_path}")
                        except Exception as e:
                            print(f"Error removing {file_path}: {e}")
                print(f"In directory '{dir_path}', kept the newest file: {newest_file}")
            else:
                print(f"No files found in directory: {dir_path}")

def main():
    while True:
        print("Cleaning directories...")
        clean_directories(TARGET_DIR)
        print("Sleeping for 60 seconds...\n")
        time.sleep(60)

if __name__ == '__main__':
    main()
