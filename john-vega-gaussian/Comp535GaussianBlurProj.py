import tensorflow as tf
import os
from PIL import Image, ImageFile
import numpy as np
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor, as_completed 
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True

def process_image_on_gpu(input_path, output_path, size, sigma):
    # Read the input image and convert it to a normalized numpy array
    image = Image.open(input_path)
    image_np = np.array(image, dtype=np.float32) / 255.0

    # Convert the image to a TensorFlow tensor
    image_tensor = tf.convert_to_tensor(image_np)

    # Generate the Gaussian kernel
    kernel = gaussian_kernel(size, sigma)

    # Apply Gaussian smoothing to the image
    smoothed_image = apply_gaussian_smoothing(image_tensor, kernel)

    # Convert the smoothed image back to a numpy array for saving
    smoothed_image_np = smoothed_image.numpy() * 255.0
    smoothed_image_np = smoothed_image_np.astype(np.uint8)

    # Save the processed image
    output_image_path = os.path.join(output_path, os.path.basename(input_path))
    Image.fromarray(smoothed_image_np).save(output_image_path)
    print(f"Processed: {input_path} -> {output_image_path}")

def process_images_in_directory(input_dir, output_dir, size, sigma):
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Gather all image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    image_paths = [os.path.join(input_dir, f) for f in image_files]

    # Process each image in parallel
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image_on_gpu, image_path, output_dir, size, sigma) for image_path in image_paths]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing image: {e}")

def gaussian_kernel(size, sigma):
    # Create a Gaussian kernel with the specified size and sigma
    center = size // 2
    x = tf.range(-center, center + 1, dtype=tf.float32)
    y = tf.range(-center, center + 1, dtype=tf.float32)
    X, Y = tf.meshgrid(x, y)
    
    kernel = (1 / (2 * np.pi * sigma**2)) * tf.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel = kernel / tf.reduce_sum(kernel)  # Normalize the kernel so that its sum is 1
    return kernel

def apply_gaussian_smoothing(image_tensor, kernel):
    # Add batch dimension to image tensor
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    
    # Expand kernel dimensions to match input channels
    kernel = tf.expand_dims(kernel, axis=-1)  # Add a channel dimension
    kernel = tf.expand_dims(kernel, axis=-1)  # Add filter dimension for depthwise convolution
    kernel = tf.tile(kernel, [1, 1, image_tensor.shape[-1], 1])  # Repeat kernel for each input channel
    
    # Perform depthwise convolution
    smoothed_image = tf.nn.depthwise_conv2d(image_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME')
    
    # Remove the batch dimension
    smoothed_image = tf.squeeze(smoothed_image, axis=0)
    return smoothed_image

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Apply Gaussian smoothing to images using TensorFlow on the GPU.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing images.")
    parser.add_argument("output_dir", type=str, help="Path to save the processed images.")
    parser.add_argument("-s", "--size", type=int, required=True, help="Size of the Gaussian kernel (must be an odd integer).")
    parser.add_argument("-g", "--sigma", type=float, required=True, help="Sigma value for Gaussian kernel.")
    args = parser.parse_args()

    # Ensure kernel size is an odd integer
    if args.size % 2 == 0:
        print("Error: The Gaussian kernel size must be an odd integer.")
        sys.exit(1)

    # Process images in the directory
    process_images_in_directory(args.input_dir, args.output_dir, args.size, args.sigma)

if __name__ == "__main__":
    main()
