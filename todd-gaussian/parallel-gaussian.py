from gaussian import *
import argparse

#Main processing function
def process_multi(input_dir, output_dir, kernel_size, sigma, display):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = os.listdir(input_dir)
    
    if not kernel_size:
        kernel_size = int(6 * sigma + 1)
        
    pool = Pool()
    argsT = input_dir, output_dir, kernel_size, sigma, display
    partial_process_image = partial(process_image, argsT=argsT)

    pool.map(partial_process_image, files)

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
    process_multi(args.input_dir, args.output_dir, args.size, args.sigma, args.display)

#python3 todd-gaussian/parallel-gaussian.py -s 5 -g 1.5 images/input-test/ images/output/
