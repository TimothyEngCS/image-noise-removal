from mpi4py import MPI
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

# Function to apply Gaussian smoothing to part of the image
def apply_gaussian_smoothing_partial(image_np, kernel):
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

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load the image only on the root process
    if rank == 0:
        image_path = '/Users/stephenreilly/Downloads/TEST/Datacluster Truck (116).jpg'
        image = Image.open(image_path)
        image_np = np.array(image)  # Convert the image to a numpy array
        img_height, img_width, channels = image_np.shape

        # Split the image into chunks (one chunk per process)
        rows_per_process = img_height // size
        chunks = [image_np[i * rows_per_process:(i + 1) * rows_per_process, :, :] for i in range(size)]
    else:
        img_height, img_width, channels = None, None, None
        chunks = None

    # Broadcast the image dimensions to all processes
    img_height, img_width, channels = comm.bcast([img_height, img_width, channels], root=0)

    # Scatter chunks of the image to all processes
    chunk = comm.scatter(chunks, root=0)

    # Define the Gaussian kernel size and standard deviation
    kernel_size = 8  # Larger kernel for stronger blur
    sigma = 4.0       # Increase sigma for more pronounced blur
    kernel = gaussian_kernel(kernel_size, sigma)

    # Apply Gaussian smoothing to the chunk
    smoothed_chunk = apply_gaussian_smoothing_partial(chunk, kernel)

    # Gather the smoothed chunks back to the root process
    smoothed_image = comm.gather(smoothed_chunk, root=0)

    # On the root process, combine the chunks and display the result
    if rank == 0:
        smoothed_image = np.vstack(smoothed_image)  # Combine the chunks
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image_np)
        plt.title('Original JPG Image')

        plt.subplot(1, 2, 2)
        plt.imshow(smoothed_image.astype(np.uint8))  # Convert back to uint8 for display
        plt.title('Blurred Image (Gaussian Smoothing)')

        plt.show()

if __name__ == '__main__':
    main()
