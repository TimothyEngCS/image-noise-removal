import cv2
import os
import sys
from multiprocessing import Pool
import argparse

# Function to apply Gaussian smoothing to a single image
def apply_gaussian_smoothing(image_info):
    input_path, output_path, sigma = image_info
    # Read the image
    image = cv2.imread(input_path)
    
    if image is not None:
        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(image, (0, 0), sigma)
        # Save the blurred image to the output path
        cv2.imwrite(output_path, blurred_image)
        print(f"Processed and saved: {output_path}")
    else:
        print(f"Could not read image: {input_path}")

def process_images(input_folder, output_folder, sigma):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Prepare image file paths for processing
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    image_info_list = []
    
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)
        image_info_list.append((input_path, output_path, sigma))

    # Use multiprocessing to apply Gaussian smoothing in parallel
    with Pool() as pool:
        pool.map(apply_gaussian_smoothing, image_info_list)

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Apply Gaussian Smoothing to Images")
    parser.add_argument("input_folder", help="Path to the folder of input images")
    parser.add_argument("sigma", type=float, help="Standard deviation for Gaussian kernel (sigma)")
    parser.add_argument("output_folder", help="Path to save the processed images")

    args = parser.parse_args()

    # Run the image processing
    process_images(args.input_folder, args.output_folder, args.sigma)
