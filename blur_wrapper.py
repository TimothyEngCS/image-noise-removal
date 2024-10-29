import os
import sys
import subprocess
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent

def run_cpp_executable(input_dir, output_dir):
    cpp_executable = "./gaussian"  # Path to your compiled C++ executable
    subprocess.run([cpp_executable, input_dir, output_dir], check=True)
    print("C++ processing complete.")

def display_images_side_by_side(input_dir, output_dir):
    input_images = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".jpg")])
    output_images = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".jpg")])
    
    if not input_images or not output_images:
        print("No images to display.")
        return

    # Preload images to improve navigation speed
    input_images_loaded = [Image.open(img) for img in input_images]
    output_images_loaded = [Image.open(img) for img in output_images]
    
    current_index = 0
    fig, ax = plt.subplots(1, 2, figsize=(12, 7))

    def update_display(index):
        ax[0].imshow(input_images_loaded[index])
        ax[0].set_title("Original", fontsize=14, pad=10)
        
        ax[1].imshow(output_images_loaded[index])
        ax[1].set_title("Blurred", fontsize=14, pad=10)
        
        for a in ax:
            a.axis('off')

        fig.tight_layout(pad=0.5)
        plt.draw()

    def on_key(event: KeyEvent):
        nonlocal current_index
        if event.key == 'right':
            current_index = (current_index + 1) % len(input_images_loaded)
        elif event.key == 'left':
            current_index = (current_index - 1) % len(input_images_loaded)
        elif event.key == 'q' or event.key == 'escape':  # Use 'q' or 'Esc' to quit
            plt.close()
            return
        update_display(current_index)

    update_display(current_index)
    fig.canvas.mpl_connect('key_press_event', on_key)
    # Use try-except to handle KeyboardInterrupt gracefully
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting...")
        plt.close()

# Specify input and output directories
input_dir = "images/test/input/"
output_dir = "images/test/output/"

# Check if correct number of arguments are provided
if len(sys.argv) == 3:
    # Get input and output directory paths from command-line arguments
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] 
    
# Run the C++ executable
run_cpp_executable(input_dir, output_dir)

# Display images side by side
display_images_side_by_side(input_dir, output_dir)
