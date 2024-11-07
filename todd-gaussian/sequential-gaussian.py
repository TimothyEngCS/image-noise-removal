from gaussian import *
import argparse

# Main processing function
def process_images(input_dir, output_dir, kernel_size, sigma, display):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    inputs = os.listdir(input_dir)

    argsT = (input_dir, output_dir, kernel_size, sigma, display)
    for filename in inputs:
        process_image(filename, argsT)

# Argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Gaussian filter to images in a directory.")
    parser.add_argument("input_dir", help="Path to the input directory containing images.")
    parser.add_argument("output_dir", help="Path to the output directory to save filtered images.")
    parser.add_argument("-s", "--size", type=int, required=True, help="Size of the Gaussian kernel.")
    parser.add_argument("-g", "--sigma", type=float, required=True, help="Sigma value for Gaussian kernel.")
    parser.add_argument("-d", "--display", action="store_true", help="Display images before and after filtering.")
    args = parser.parse_args()

    
    inputs = os.listdir(args.input_dir)
    num_files = 0
    for filename in inputs:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            num_files += 1
    
    process_images(args.input_dir, args.output_dir, args.size, args.sigma, args.display)

#python3 todd-gaussian/sequential-gaussian.py -s 5 -g 1.5 images/input-test/ images/output/
