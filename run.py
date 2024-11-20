from transformers import AutoImageProcessor, AutoModelForVideoClassification
from videoloader import VideoDataset, SSv2, HMDB, UCF
import torch
import cv2

# Load data

dataloader = VideoDataset(HMDB())

# Load model and processor
processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = AutoModelForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

for elem in dataloader:
    support_set = elem["support_set"]
    target_set = elem["target_set"]
    target_labels = elem["target_labels"]

    # Visualize support
    # TODO THE LABEL OF THE SUPPORTS IS HIGHLY WRONG!
    support_set_flat_labels = [dataloader.class_folders[int(x.item())] for x in elem["support_labels"]]
    support_set_flat = support_set.reshape(5, 5, 8, 3, 224, 224)
    counter = 0
    for k in support_set_flat:
        for n in k:
            print(support_set_flat_labels[counter])
            for i in n:
                cv2.imshow("image", i.permute(1, 2, 0).numpy())
                cv2.waitKey(0)
            counter += 1

    # Process data
    inputs = processor(torch.unbind(support_set), return_tensors="pt")

    processor