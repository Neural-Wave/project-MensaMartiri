import pandas as pd
import json
from dataloader import *
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import cv2
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

def visualize_images(data_loader, num_images=4):

    images, labels = next(iter(data_loader))
    
    images = images[:num_images]
    labels = labels[:num_images]

    fig, axs = plt.subplots(1, num_images, figsize=(15, 5))
    
    for i in range(num_images):

        img = np.transpose(images[i].numpy(), (1, 2, 0))
        
        # Display the image
        axs[i].imshow(img)
        axs[i].axis('off') 

        label_str = f"Label: {labels[i].numpy()}" 
        axs[i].set_title(label_str)

    plt.tight_layout()
    plt.show()



def get_img_path(path, target_folder = "dtp-sbm-segmentation-video-tasks-bars-stopper-alignment-images-hackaton-usi"):
    start_idx = path.find(target_folder)
    if start_idx != -1:
        subsequent_folders = path[start_idx + len(target_folder):]
    return subsequent_folders


def check_all_images_same_size():
    # Load the CSV file
    directory = config.DATASET_PATH

    # Get all images from the directory recursively on subdirectories
    all_images = list(Path(directory + "/example_set").rglob("*.jpg")) 
    sizes = [Image.open(img).size for img in all_images]

    # Check if all sizes are the same
    all_same = all(size == sizes[0] for size in sizes)
    return all_same, sizes[0]


def halve_image(image):
    width, height = image.size
    new_width = width // 2
    
    # Crop the image to keep only the right half
    cropped_image = image.crop((new_width, 0, width, height))
    return cropped_image


def _get_flattened_size(conv_layers, input_tensor):
    with torch.no_grad():
        output = conv_layers(input_tensor)
        return output.numel() 

def parse_img_metadata(img_metadata_str):

    if img_metadata_str in ["", "nan"]:
        img_metadata = []
    else:
        try:
            img_metadata = json.loads(img_metadata_str)
            if not isinstance(img_metadata, list):
                img_metadata = []
                print("Invalid format for metadata")
        except (json.JSONDecodeError, TypeError):
            img_metadata = []

    return img_metadata


def to_coordinates_and_labels(img_metadata_serial):
    img_metadata = parse_img_metadata(img_metadata_serial)

    coordinates = [
        [
            metadata.get('x', 0) * metadata.get('original_width', 0) / 100, 
            metadata.get('y', 0) * metadata.get('original_height', 0) / 100,
            int(metadata.get('keypointlabels', ['']) == ['Contact'])
        ] 
            for metadata in img_metadata if isinstance(metadata, dict)
    ]
    return np.array(coordinates)

def to_label(img_metadata_serial):
    img_metadata = parse_img_metadata(img_metadata_serial)

    contact_labels = [
        metadata.get("keypointlabels") == ["Contact"]
        for metadata in img_metadata if isinstance(metadata, dict)
    ]

    return int(all(contact_labels) or not contact_labels)

def patch_to_label(img_metadata_serial):
    img_metadata = parse_img_metadata(img_metadata_serial)

    contact_labels = [
        metadata.get("keypointlabels") == ["Contact"]
        for metadata in img_metadata if isinstance(metadata, dict)
    ]

    return int(any(contact_labels)) 

def single_keypoints_to_label(keypoints):
    img_metadata = utils.parse_img_metadata(keypoints)
    return list(map(lambda kp_metadata: int(kp_metadata['keypointlabels'] == ['Contact']), img_metadata))


def extract_patch(image, x, y, length=50):
    """
    Extracts a patch of size length x length centered at (x, y).
    """
    x = int(x)
    y = int(y)
    return image.crop((x - length//2, y - length//2, x + length//2, y + length//2))


def plot_patch(patch):
    plt.imshow(patch)
    plt.scatter(patch.size[0]//2, patch.size[1]//2, c='r')
    plt.axis('off')
    plt.show()


def normalize_brightness(image, target_brightness=0.5):

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mean_brightness = np.mean(hsv[:, :, 2] / 255.0)
    scale = target_brightness / mean_brightness
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * scale, 0, 255).astype(np.uint8)
    normalized_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return normalized_image

def normalize_brightness_PIL(image: Image.Image, target_brightness=0.5) -> Image.Image:

    image_np = np.array(image)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    mean_brightness = np.mean(hsv[:, :, 2] / 255.0)
    scale = target_brightness / mean_brightness
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * scale, 0, 255).astype(np.uint8)
    normalized_image_np = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    normalized_image = Image.fromarray(normalized_image_np)

    return normalized_image

def histogram_eq_global(img):
    # Convert RGB image to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Extract the V channel
    v_channel = img_hsv[:, :, 2]
    
    # Compute histogram of the V channel
    hist, _ = np.histogram(v_channel, bins=256, range=(0, 256))
    
    # Compute cumulative density function (CDF)
    cdf = hist.cumsum()
    
    # Normalize CDF to range [0, 255]
    cdf_normalized = np.clip((cdf - cdf.min()) * 255 / (cdf.max() - cdf.min()), 0, 255).astype(np.uint8)
    
    # Map the V channel values to equalized values using the CDF
    v_eq = cdf_normalized[v_channel]
    img_hsv[:, :, 2] = v_eq

    # Convert the image back to RGB
    eq_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    return eq_img

def create_test_set_csv():

    path = config.DATASET_PATH + '/example_set'

    image_paths = []
    labels = []

    aligned_path = os.path.join(path, 'aligned')
    for root, _, files in os.walk(aligned_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):  
                image_paths.append(os.path.join(root, file))
                labels.append(1)  


    not_aligned_path = os.path.join(path, 'not_aligned')
    for root, _, files in os.walk(not_aligned_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')): 
                image_paths.append(os.path.join(root, file))
                labels.append(0) 


    df = pd.DataFrame({'image': image_paths, 'label': labels})
    df['image'] = df['image'].str.replace(r'../../../', '/data/local-files/?d=', regex=True)

    # Save the DataFrame to a CSV file
    csv_file_path = os.path.join(path, config.TEST_SET_PATH)
    df.to_csv(csv_file_path, index=False)

    print(f"CSV file saved at: {csv_file_path}")


def get_test_data(test_set_directory=config.DATASET_PATH + '/example_set/'):
    data = []
    for root, directories, files in os.walk(test_set_directory):
        for file in files:
            # Check if the parent directory is 'aligned'
            label = int(os.path.basename(root) == 'aligned')

            if file.endswith('.jpg'):
                img_path = os.path.join(root, file)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                data.append((image, label, img_path))
                
    return data


def plot_confusion_matrix(pre_dictions, labels):

    cm = confusion_matrix(labels, pre_dictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Aligned', 'Aligned'], yticklabels=['Not Aligned', 'Aligned'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('plots/confusion_matrix.png')
    plt.show()