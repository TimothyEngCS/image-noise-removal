import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 

# Function to create a Gaussian kernel
def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size), np.float32)
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            diff_x = i - center
            diff_y = j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(diff_x**2 + diff_y**2) / (2 * sigma**2))
    
    # Normalize the kernel so that its sum is 1
    return kernel / np.sum(kernel)

# Function to apply the Gaussian filter by convolution
def apply_gaussian_smoothing(image_np, kernel):
    kernel_size = kernel.shape[0]
    img_height, img_width = image_np.shape[:2]
    
    # Add padding to the image for convolution
    pad_size = kernel_size // 2
    padded_image = np.pad(image_np, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
    
    smoothed_image = np.zeros_like(image_np, dtype=np.float32)
    
    # Convolve the image with the Gaussian kernel
    for i in range(img_height):
        for j in range(img_width):
            for k in range(image_np.shape[2]):  # Loop over color channels (for color images)
                region = padded_image[i:i + kernel_size, j:j + kernel_size, k]
                smoothed_image[i, j, k] = np.sum(region * kernel)
    
    return smoothed_image

# Main function
def main():
    # Load the JPG image file using Pillow
    image_path = '/Users/stephenreilly/Downloads/TEST/Datacluster Truck (116).jpg'
    image = Image.open(image_path)
    image_np = np.array(image)  # Convert the image to a numpy array

    # Define the Gaussian kernel size and standard deviation
    kernel_size = 3  # Larger kernel for stronger blur
    sigma = 4.0       # Increase sigma for more pronounced blur

    # Generate the Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)

    # Apply Gaussian smoothing to the image
    smoothed_image = apply_gaussian_smoothing(image_np, kernel)

    # Display the original and smoothed images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title('Original JPG Image')

    plt.subplot(1, 2, 2)
    plt.imshow(smoothed_image.astype(np.uint8))  # Convert back to uint8 for display
    plt.title('Blurred Image (Gaussian Smoothing)')

    plt.show()

# Call the main function
if __name__ == '__main__':
    main()
