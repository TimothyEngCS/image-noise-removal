# THIS VERSION OF THE GAUSSIAN KERNEL WAS REFACTORED TO RUN UTILIZING THE GPU INSTEAD OF CPU. The numpy image was converted to a tensor a 
# A depthwise convolutional layer was also added in order to 
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import numpy as np
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
ImageFile.LOAD_TRUNCATED_IMAGES = True
def process_images_on_gpu(input_path, output_path, sigma):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    images = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    image_paths = [os.path.join(input_path, f) for f in images]

    start_time = time.time()

    for image_path in image_paths:
         individual_image_time = time.time()
         image = Image.open(image_path)
         image_np = np.array(image, dtype=np.float32) / 255.0

         image_tensor = tf.convert_to_tensor(image_np)

         sigma = 4.0

         kernel = gaussian_kernel(sigma)

         smoothed_image = apply_gaussian_smoothing(image_tensor, kernel)


         if not os.path.exists(output_path):
             os.makedirs(output_path)
 
         output_image_path = os.path.join(output_path, os.path.basename(image_path))
         smoothed_image = Image.fromarray(np.uint8(smoothed_image))
         smoothed_image.save(output_image_path)
         end_image_time = time.time()
         image_time = end_image_time - individual_image_time
         print(f"Processed {image_path} in {image_time:.2f} seconds")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Processed all data and stored in {total_time:.2f} seconds") 

# Function to create a Gaussian kernel using TensorFlow
def gaussian_kernel(sigma):
    size = int(2 * np.ceil(3 * sigma) + 1)
    center = size // 2
    x = tf.range(-center, center + 1, dtype=tf.float32)
    y = tf.range(-center, center + 1, dtype=tf.float32)
    X, Y = tf.meshgrid(x, y)
    
    kernel = (1 / (2 * np.pi * sigma**2)) * tf.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel = kernel / tf.reduce_sum(kernel)  # Normalize the kernel so that its sum is 1
    return kernel

# Function to apply the Gaussian filter using TensorFlow convolution
def apply_gaussian_smoothing(image_tensor, kernel):
    # Add batch dimension
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    
    # Expand kernel to match the number of input channels
    kernel = tf.expand_dims(kernel, axis=-1)  # Add a channel dimension
    kernel = tf.expand_dims(kernel, axis=-1)  # Add filter dimension for depthwise convolution
    kernel = tf.tile(kernel, [1, 1, image_tensor.shape[-1], 1])  # Repeat kernel for each input channel

    # Perform depthwise convolution (applies the same filter to each input channel)
    smoothed_image = tf.nn.depthwise_conv2d(image_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME')
    
    # Remove the batch dimension
    smoothed_image = tf.squeeze(smoothed_image, axis=0)
    return smoothed_image

# Main function
def main():
    # Load the JPG image file using Pillow
    root = os.getcwd()
    image_path = os.path.join(root, 'COMP535/archive/Truck/Datacluster Truck (116).jpg')
    start_time = time.time()
    image = Image.open(image_path)
    image_np = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]

    # Convert the image to a TensorFlow tensor
    image_tensor = tf.convert_to_tensor(image_np)

    # Define the Gaussian kernel size and standard deviation
    kernel_size = 32  # Larger kernel for stronger blur
    sigma = 4.0       # Increase sigma for more pronounced blur

    # Generate the Gaussian kernel using TensorFlow
    kernel = gaussian_kernel(sigma)

    # Apply Gaussian smoothing to the image using TensorFlow operations
    smoothed_image = apply_gaussian_smoothing(image_tensor, kernel)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Processed {image_path} in {total_time:.2f} seconds")

    #smoothed_image = process_image(image_path, sigma)

    # Convert the smoothed image back to a NumPy array for visualization
    smoothed_image_np = smoothed_image.numpy() * 255.0
    smoothed_image_np = smoothed_image_np.astype(np.uint8)

    # Display the original and smoothed images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title('Original JPG Image')

    plt.subplot(1, 2, 2)
    
    plt.imshow(smoothed_image_np)
    
    plt.title('Blurred Image (Gaussian Smoothing)')

    plt.show()

    # Display the Gaussian kernel in 2D
    plt.figure(figsize=(6, 6))
    plt.imshow(kernel.numpy(), cmap='viridis')
    plt.title('Gaussian Kernel (2D)')
    plt.colorbar()
    plt.show()

    # Display the Gaussian kernel in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0, kernel.shape[0], 1)
    y = np.arange(0, kernel.shape[1], 1)
    X, Y = np.meshgrid(x, y)
    Z = kernel.numpy()  # Convert TensorFlow tensor to NumPy array
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title('Gaussian Kernel (3D)')
    plt.show()

    plt.show()
    input_path = os.path.join(root, 'COMP535/archive/Truck')
    output_path = os.path.join(root, 'COMP535/archive/GPUTruckOutput')
    process_images_on_gpu(input_path, output_path, sigma)
# Call the main function
if __name__ == '__main__':
    main()
