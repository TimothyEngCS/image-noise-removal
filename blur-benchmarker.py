import os
import time
import argparse
import subprocess
from PIL import Image
import matplotlib.pyplot as plt

def display_images_side_by_side(input_dir, output_dir):
    input_images = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".jpg")])
    output_images = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".jpg")])
    
    if not input_images or not output_images:
        print("No images to display.")
        return

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

    def on_key(event):
        nonlocal current_index
        if event.key == 'right':
            current_index = (current_index + 1) % len(input_images_loaded)
        elif event.key == 'left':
            current_index = (current_index - 1) % len(input_images_loaded)
        elif event.key == 'q' or event.key == 'escape':
            plt.close()
            return
        update_display(current_index)

    update_display(current_index)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    
def run_with_timer(command, description="Script", is_cpp=False, args=None):
    print(f"Running {description}...")

    # Compile C++ code if it's a C++ command
    if is_cpp:
        # Use `description` to name the executable dynamically
        exec_path = f"./{description}/{description}_executable"
        compile_cmd = ["g++-14", "-std=c++17", "-fopenmp", command, "-o", exec_path]
        
        try:
            subprocess.run(compile_cmd, check=True)
            print("Compilation successful.")
        except subprocess.CalledProcessError:
            print(f"Error: Compilation of {description} failed.")
            return None

        # Start timing after compilation
        start_time = time.time()
        # Use the executable path based on `description`
        run_cmd = [exec_path, args.input_path, args.output_path, str(args.size), str(args.sigma)]
    else:
        # Start timing for Python script execution
        start_time = time.time()
        run_cmd = command

    try:
        subprocess.run(run_cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"Error: {description} did not execute successfully.")
        return None

    elapsed_time = time.time() - start_time
    print(f"{description} completed in {elapsed_time:.2f} seconds.")
    return elapsed_time

def main():
    parser = argparse.ArgumentParser(description="Run specified implementation(s) with a timer.")
    parser.add_argument("input_path", help="Path to the input directory.")
    parser.add_argument("output_path", help="Path to the output directory.")
    parser.add_argument("-i", "--implementation", help="The implementation to run (e.g., todd-gaussian). If not specified, all implementations are run.")
    parser.add_argument("-s", "--size", type=int, required=True, help="Size parameter to pass to each implementation.")
    parser.add_argument("-g", "--sigma", type=float, required=True, help="Sigma parameter to pass to each implementation.")
    parser.add_argument("--display", action="store_true", help="Display images after processing.")
    args = parser.parse_args()

    # Define script paths for each implementation, with 'stephen-gaussian' as a C++ file
    scripts = {
        "stephen-gaussian": "stephen-gaussian/gaussian.cpp", 
        "timothy-gaussian": "timothy-gaussian/gaussian-parallel.py",
        "john-vega-gaussian": "john-vega-gaussian/Comp535GaussianBlurProj.py",
        "michael-gaussian": "michael-gaussian/PhotoGaussianVF.py",
        "todd-gaussian": "todd-gaussian/parallel-gaussian.py",
        "sequential-gaussian": "todd-gaussian/sequential-gaussian.py"
    }

    # Determine which scripts to run based on the implementation argument
    if args.implementation:
        if args.implementation not in scripts:
            print(f"Error: Implementation '{args.implementation}' not found.")
            print("Available implementations:", ", ".join(scripts.keys()))
            sys.exit(1)
        implementations_to_run = {args.implementation: scripts[args.implementation]}
    else:
        implementations_to_run = scripts

    # Ensure input and output directories exist
    os.makedirs(args.input_path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    for name, script_path in implementations_to_run.items():
        output_dir_for_implementation = os.path.join(args.output_path, name)
        os.makedirs(output_dir_for_implementation, exist_ok=True)
        
        is_cpp = script_path.endswith(".cpp")
        if is_cpp:
            # Pass the script_path for C++ compilation and name as the run command, along with args
            run_with_timer(script_path, description=name, is_cpp=True, args=args)
        else:
            # Run Python code
            command = [
                "python3", script_path, args.input_path, output_dir_for_implementation,
                "-s", str(args.size), "-g", str(args.sigma)
            ]
            run_with_timer(command, description=name, args=args)

        # Display images if the display flag is set and a specific implementation is provided
        if args.display and args.implementation:
            display_images_side_by_side(args.input_path, output_dir_for_implementation)

if __name__ == "__main__":
    main()
