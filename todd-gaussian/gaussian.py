import os
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Kernel generation function
def gaussianKernel(size, sigma):
    kernel = np.zeros((size, size), np.float32)
    m = size // 2
    n = size // 2

    for x in range(-m, m + 1):
        for y in range(-n, n + 1):
            x1 = 2 * np.pi * (sigma ** 2)
            x2 = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            kernel[x + m, y + n] = (1 / x1) * x2
    return kernel

# Convolution function
def convolution(oldimage, kernel):
    image_h, image_w = oldimage.shape[:2]
    kernel_h, kernel_w = kernel.shape

    # Padding based on image dimensions
    if len(oldimage.shape) == 3:
        image_pad = np.pad(oldimage, ((kernel_h // 2, kernel_h // 2), (kernel_w // 2, kernel_w // 2), (0, 0)), 'constant', constant_values=0).astype(np.float32)
    else:
        image_pad = np.pad(oldimage, ((kernel_h // 2, kernel_h // 2), (kernel_w // 2, kernel_w // 2)), 'constant', constant_values=0).astype(np.float32)
    
    h, w = kernel_h // 2, kernel_w // 2
    image_conv = np.zeros(image_pad.shape)
    
    for i in range(h, image_pad.shape[0] - h):
        for j in range(w, image_pad.shape[1] - w):
            x = image_pad[i - h:i - h + kernel_h, j - w:j - w + kernel_w]
            x = x.flatten() * kernel.flatten()
            image_conv[i][j] = x.sum()
    
    h_end, w_end = -h, -w
    if h == 0:
        return image_conv[h:, w:w_end]
    if w == 0:
        return image_conv[h:h_end, w:]
    return image_conv[h:h_end, w:w_end]

# Apply Gaussian filter
def gaussianFilter(image_path, kernel_size, sigma):
    image = Image.open(image_path)
    image = np.asarray(image)

    gaussian_kernel = gaussianKernel(kernel_size, sigma)
    im_filtered = np.zeros_like(image, dtype=np.float32)

    # Apply convolution for each channel
    for c in range(3):
        im_filtered[:, :, c] = convolution(image[:, :, c], gaussian_kernel)

    return im_filtered.astype(np.uint8)

# Main processing function to handle directories
def process_images(input_dir, output_dir, kernel_size, sigma, display):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            print(f"Processing {image_path}...")

            # Apply Gaussian filter
            filtered_image = gaussianFilter(image_path, kernel_size, sigma)

            # Save the filtered image
            output_path = os.path.join(output_dir, filename)
            Image.fromarray(filtered_image).save(output_path)
            print(f"Saved filtered image to {output_path}")

            # Display the image if display flag is set
            if display:
                before = Image.open(image_path)
                after = Image.open(output_path)
                
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(before)
                ax[0].set_title("Original")
                ax[1].imshow(after)
                ax[1].set_title("Filtered")
                
                for a in ax:
                    a.axis("off")
                plt.show()

# Argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Gaussian filter to images in a directory.")
    parser.add_argument("input_dir", help="Path to the input directory containing images.")
    parser.add_argument("output_dir", help="Path to the output directory to save filtered images.")
    parser.add_argument("-s", "--size", type=int, required=True, help="Size of the Gaussian kernel.")
    parser.add_argument("-g", "--sigma", type=float, required=True, help="Sigma value for Gaussian kernel.")
    parser.add_argument("-d", "--display", action="store_true", help="Display images before and after filtering.")
    args = parser.parse_args()

    # Process images with the specified parameters
    process_images(args.input_dir, args.output_dir, args.size, args.sigma, args.display)


#python3 todd-gaussian/gaussian.py images/input-test/ images/output/ -s 5 -g 1.5