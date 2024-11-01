import os
import cv2
import numpy as np
import multiprocessing
import argparse

def process_image(args):
    input_path, output_path, sigma = args
    try:
        # Read the input image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Could not read image {input_path}")
            return

        # Apply Gaussian smoothing
        smoothed_image = cv2.GaussianBlur(image, (0, 0), sigma)

        # Save the processed image to the output directory
        output_filename = os.path.join(output_path, os.path.basename(input_path))
        cv2.imwrite(output_filename, smoothed_image)
        print(f"Processed: {input_path} -> {output_filename}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def main(input_folder, sigma, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Gather all image files in the input directory
    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # Set up arguments for multiprocessing
    arguments = [(input_path, output_folder, sigma) for input_path in image_files]

    # Process images in parallel using multiprocessing
    with multiprocessing.Pool() as pool:
        pool.map(process_image, arguments)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Gaussian Smoothing to 2D image files in parallel.")
    parser.add_argument("input_folder", type=str, help="Path to the input image folder")
    parser.add_argument("sigma", type=float, help="Standard deviation for Gaussian kernel")
    parser.add_argument("output_folder", type=str, help="Path to save the processed images")
    
    args = parser.parse_args()
    main(args.input_folder, args.sigma, args.output_folder)
