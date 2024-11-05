import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import argparse

def process_image(image_path, output_path, sigma):
    """
    Processes a single image by applying Gaussian smoothing and saves it.
    """
    start_time = time.time()  # Start timer for this image

    image = Image.open(image_path)
    image_array = np.array(image)
    
    # If the image is in RGB format, apply the filter to each channel
    if image_array.ndim == 3:  # Check if it's an RGB image
        smoothed_image_array = np.zeros_like(image_array)
        for channel in range(3):  # Apply filter to each channel independently
            smoothed_image_array[:, :, channel] = gaussian_filter(image_array[:, :, channel], sigma=sigma)
    else:
        # If grayscale, just apply the filter directly
        smoothed_image_array = gaussian_filter(image_array, sigma=sigma)
    
    smoothed_image = Image.fromarray(np.uint8(smoothed_image_array))
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    output_image_path = os.path.join(output_path, os.path.basename(image_path))
    smoothed_image.save(output_image_path)

    end_time = time.time()  # End timer for this image
    duration = end_time - start_time  # Calculate the time taken
    print(f"Processed {image_path} in {duration:.2f} seconds")  # Print the time taken for each image

    return output_image_path

def process_images_in_parallel(input_path, output_path, sigma):
    """
    Processes images in parallel using ThreadPoolExecutor.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    image_paths = [os.path.join(input_path, f) for f in image_files]

    start_time = time.time()  # Start timer for the entire process
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, image_path, output_path, sigma): image_path for image_path in image_paths}
        for future in as_completed(futures):
            image_path = futures[future]
            try:
                result = future.result()
                print(f"Processed {image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    end_time = time.time()  # End timer for the entire process
    total_duration = end_time - start_time  # Calculate total time taken
    print(f"Processed all images in {total_duration:.2f} seconds")  # Print total time
 

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Apply Gaussian filter to images in a directory.")
    parser.add_argument("input_path", help="Path to the input directory containing images.")
    parser.add_argument("output_path", help="Path to the output directory for saving images.")
    # 'size' is kept for consistency but not used directly
    parser.add_argument("-s", "--size", type=int, required=True, help="Size of the Gaussian kernel (not used directly).")
    parser.add_argument("-g", "--sigma", type=float, required=True, help="Sigma value for Gaussian kernel.")
    args = parser.parse_args()

    # Process images
    process_images_in_parallel(args.input_path, args.output_path, args.sigma)

if __name__ == "__main__":
    main()
