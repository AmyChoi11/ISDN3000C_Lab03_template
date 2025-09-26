# Lab 3: RDK X5 AI Vision Tasks

This repository contains my work for the Lab 3 assignment.

## Basic Task

The basic task involved running a classifier on a single image and displaying the result on a webpage hosted by the RDK. The `classify.py` script successfully processed the `sample_image.png` and classified it using a pre-trained ResNet18 model.

### Results
- **Script Used**: `classify.py`
- **Image Processed**: `sample_image.png` 
- **AI Model**: ResNet18 (pre-trained on ImageNet)
- **Classification Result**: The AI successfully identified the image with high confidence

A screenshot of the webpage task should be saved as `result.png` (to be added after completing the web server task).

## Advanced Task: Batch Image Classification

The advanced task involved creating a Python script (`classify_batch.py`) that processes an entire folder of images and outputs the classification results into a structured CSV file.

- **Script**: `classify_batch.py`
- **How to Run**: `python classify_batch.py` (processes images in the `task2` directory)
- **Output**: The script generates `results.csv` with three columns: `image_name`, `detected_class`, and `confidence_level`

### Result Summary

The script was run on a folder containing 10 images across multiple classes. The AI model successfully classified various objects including:
- Goldfish (98.45% and 96.87% confidence)
- Axolotl (100.00% and 99.99% confidence) 
- Jellyfish (99.97% confidence)
- Japanese Chin dogs (99.66% and 83.53% confidence)
- Espresso (97.39% and 95.92% confidence)
- One ambiguous image classified as joystick (7.59% confidence)

The full, machine-readable output is available in the `results.csv` file included in this repository.

## Repository Structure

```
├── classify.py              # Single image classification script
├── classify_batch.py        # Batch image classification script
├── requirements.txt         # Python dependencies
├── sample_image.png         # Sample image for testing
├── results.csv             # Batch processing results
├── task1/                  # Web server directory
│   ├── index.html          # AI showcase webpage
│   └── sample_image.png    # Image copy for web display
├── task2/                  # Directory containing test images
└── README.md               # This file
```

## How to Use

1. **Single Image Classification**:
   ```bash
   python classify.py
   ```

2. **Batch Image Classification**:
   ```bash
   python classify_batch.py
   ```

3. **Web Showcase** (from task1 directory):
   ```bash
   cd task1
   python -m http.server
   ```
   Then visit `http://192.168.127.10:8000` from your laptop browser.

## Technologies Used

- **Python**: Main programming language
- **PyTorch**: Deep learning framework
- **ResNet18**: Pre-trained convolutional neural network
- **PIL (Pillow)**: Image processing library
- **CSV**: For structured data output
- **HTML/CSS**: Web presentation layer