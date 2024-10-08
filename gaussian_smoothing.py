import numpy as np
import matplotlib.pyplot as plt
import struct

#python image library - PIL


# Function to read a 24-bit BMP file
def read_bmp(file_path):
    with open(file_path, 'rb') as f:
        # Read BMP header
        header = f.read(14)  # BMP file header is 14 bytes
        dib_header = f.read(40)  # DIB header (typically 40 bytes for BMP)
        
        # Extract width and height from DIB header
        width = struct.unpack('I', dib_header[4:8])[0]
        height = struct.unpack('I', dib_header[8:12])[0]
        bpp = struct.unpack('H', dib_header[14:16])[0]  # Bits per pixel
        
        if bpp != 24:
            raise ValueError("Only 24-bit BMP images are supported")

        # Read pixel data (BMP stores pixel data bottom-up)
        image_data = np.zeros((height, width, 3), dtype=np.uint8)
        padding = (4 - (width * 3 % 4)) % 4  # BMP row padding to a multiple of 4 bytes

        for i in range(height - 1, -1, -1):  # Read rows from bottom to top
            row_data = f.read(width * 3)
            row = np.frombuffer(row_data, dtype=np.uint8).reshape(width, 3)
            
            # BMP stores colors as BGR, so we need to convert them to RGB
            image_data[i, :, :] = row[:, [2, 1, 0]]  # BGR to RGB
            
            f.read(padding)  # Skip padding bytes if necessary

    return image_data

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
def apply_gaussian_smoothing(image, kernel):
    kernel_size = kernel.shape[0]
    img_height, img_width = image.shape[:2]
    
    # Add padding to the image for convolution
    pad_size = kernel_size // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
    
    smoothed_image = np.zeros_like(image, dtype=np.float32)
    
    # Convolve the image with the Gaussian kernel
    for i in range(img_height):
        for j in range(img_width):
            for k in range(image.shape[2]):  # Loop over color channels (for color images)
                region = padded_image[i:i + kernel_size, j:j + kernel_size, k]
                smoothed_image[i, j, k] = np.sum(region * kernel)
    
    return smoothed_image

# Main function
def main():
    # Load the BMP image file
    image_path = '/Users/stephenreilly/Desktop/USC/CSCI 103/cs103_vm/projects/project3/usc_ucla_wikimedia.bmp'
    image_np = read_bmp(image_path)

    # Define the Gaussian kernel size and standard deviation
    kernel_size = 5  # 5x5 kernel
    sigma = 1.0

    # Generate the Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)

    # Apply Gaussian smoothing to the image
    smoothed_image = apply_gaussian_smoothing(image_np, kernel)

    # Display the original and smoothed images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)

    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(smoothed_image.astype(np.uint8))  # Convert back to uint8 for display
    plt.title('Blur Image ')

    plt.show()

# Call the main function
if __name__ == '__main__':
    main()
