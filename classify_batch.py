# classify_batch.py
#
# This script runs a pre-trained PyTorch AI model (ResNet18) to classify
# multiple images in a directory. It is designed for the RDK X5 AI Vision Challenge.
#
# Steps it performs:
# 1. Loads a pre-trained image classification model.
# 2. Downloads the list of human-readable labels (ImageNet classes).
# 3. Finds all image files in the specified directory.
# 4. Loads and pre-processes each image.
# 5. Performs inference (makes a prediction) for each image.
# 6. Prints the prediction and confidence score for each image.

import torch
from torchvision import models, transforms
from PIL import Image
import json
import urllib.request
import os
import glob
import csv

# --- Configuration ---
MODEL_NAME = "ResNet18"
IMAGES_DIRECTORY = "task2"  # Directory containing images to process
# URL to a raw JSON file containing the 1000 ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"


def get_image_files(directory):
    """
    Returns a list of image files in the specified directory.
    """
    # Common image extensions
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.gif']
    image_files = []
    
    for extension in image_extensions:
        pattern = os.path.join(directory, extension)
        image_files.extend(glob.glob(pattern))
    
    return image_files


def get_model():
    """
    Loads and returns a pre-trained ResNet18 model in evaluation mode.
    The first time this runs, it will download the model weights.
    """
    print(f"Loading pre-trained model: {MODEL_NAME}...")
    # Load a model pre-trained on the ImageNet dataset
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Set the model to evaluation mode. This is important for inference.
    model.eval()
    print("Model loaded successfully.")
    return model

def get_labels():
    """
    Downloads and returns the list of ImageNet class labels.
    """
    print(f"Downloading class labels from {LABELS_URL}...")
    with urllib.request.urlopen(LABELS_URL) as url:
        labels = json.loads(url.read().decode())
    print("Labels downloaded successfully.")
    return labels

def process_image(image_path):
    """
    Loads an image and applies the necessary transformations for the model.
    """
    print(f"Processing image: {image_path}...")
    # Transformations must match what the model was trained on.
    # 1. Resize to 256x256
    # 2. Center crop to 224x224
    # 3. Convert to a PyTorch Tensor
    # 4. Normalize with ImageNet's mean and standard deviation
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Open the image file
    img = Image.open(image_path).convert('RGB')
    
    # Apply the transformations
    img_t = preprocess(img)
    
    # The model expects a batch of images, so we add a "batch" dimension of 1.
    # [3, 224, 224] -> [1, 3, 224, 224]
    batch_t = torch.unsqueeze(img_t, 0)
    print("Image processed.")
    return batch_t


def predict(model, image_tensor, labels):
    """
    Performs inference and returns the top prediction.
    """
    print("Running AI inference...")
    # Perform inference without calculating gradients
    with torch.no_grad():
        output = model(image_tensor)

    # The output contains raw scores (logits). We apply a softmax function
    # to convert these scores into probabilities.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the top 1 prediction
    top1_prob, top1_cat_id = torch.topk(probabilities, 1)
    
    # Look up the category name from the labels list
    category_name = labels[top1_cat_id.item()]
    confidence_score = top1_prob.item() * 100
    
    print("Inference complete.")
    return category_name, confidence_score


if __name__ == "__main__":
    try:
        # Execute the main steps
        model = get_model()
        labels = get_labels()
        
        # Get all image files from the directory
        image_files = get_image_files(IMAGES_DIRECTORY)
        
        if not image_files:
            print(f"\n[ERROR] No image files found in directory '{IMAGES_DIRECTORY}'.")
            print("Please make sure there are image files (png, jpg, jpeg, bmp, tiff, gif) in the directory.")
        else:
            print(f"\nFound {len(image_files)} image(s) to process.\n")
            
            # Prepare CSV file
            csv_filename = "results.csv"
            results = []
            
            # Process each image
            for i, image_path in enumerate(image_files, 1):
                print(f"=== Processing Image {i}/{len(image_files)} ===")
                print(f"File: {os.path.basename(image_path)}")
                
                try:
                    image_tensor = process_image(image_path)
                    category, confidence = predict(model, image_tensor, labels)
                    
                    # Store result for CSV
                    results.append({
                        'image_name': os.path.basename(image_path),
                        'detected_class': category,
                        'confidence_level': f"{confidence:.2f}%"
                    })
                    
                    # Print the result for this image
                    print(f"Prediction: {category}, with {confidence:.2f}% confidence.")
                    print("-" * 50)
                    
                except Exception as e:
                    print(f"[ERROR] Failed to process {os.path.basename(image_path)}: {e}")
                    # Store error result for CSV
                    results.append({
                        'image_name': os.path.basename(image_path),
                        'detected_class': 'ERROR',
                        'confidence_level': '0.00%'
                    })
                    print("-" * 50)
            
            # Write results to CSV file
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['image_name', 'detected_class', 'confidence_level']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header
                writer.writeheader()
                
                # Write all results
                writer.writerows(results)
            
            print(f"\n=== Batch Processing Complete ===")
            print(f"Results saved to '{csv_filename}'")
            print(f"Processed {len(results)} images successfully.")

    except FileNotFoundError:
        print(f"\n[ERROR] The directory '{IMAGES_DIRECTORY}' was not found.")
        print("Please make sure the directory exists and contains image files.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        print("Please check your internet connection and that all libraries are installed correctly.")