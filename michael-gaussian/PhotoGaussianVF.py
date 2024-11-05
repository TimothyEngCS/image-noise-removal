import os
import cv2
import numpy as np
import multiprocessing
import argparse

def process_image(args):
    input_path, output_path, size, sigma = args
    try:
        # Read the input image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Could not read image {input_path}")
            return

        # Apply Gaussian smoothing with specified kernel size and sigma
        smoothed_image = cv2.GaussianBlur(image, (size, size), sigma)

        # Save the processed image to the output directory
        output_filename = os.path.join(output_path, os.path.basename(input_path))
        cv2.imwrite(output_filename, smoothed_image)
        print(f"Processed: {input_path} -> {output_filename}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def main(input_dir, output_dir, size, sigma):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Gather all image files in the input directory
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # Set up arguments for multiprocessing
    arguments = [(input_path, output_dir, size, sigma) for input_path in image_files]

    # Process images in parallel using multiprocessing
    with multiprocessing.Pool() as pool:
        pool.map(process_image, arguments)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Gaussian Smoothing to images in parallel.")
    parser.add_argument("input_dir", type=str, help="Path to the input image directory")
    parser.add_argument("output_dir", type=str, help="Path to save the processed images")
    parser.add_argument("-s", "--size", type=int, required=True, help="Size of the Gaussian kernel (must be an odd integer).")
    parser.add_argument("-g", "--sigma", type=float, required=True, help="Sigma value for Gaussian kernel.")
    
    args = parser.parse_args()

    # Ensure kernel size is an odd integer
    if args.size % 2 == 0:
        print("Error: The Gaussian kernel size must be an odd integer.")
        sys.exit(1)

    main(args.input_dir, args.output_dir, args.size, args.sigma)
