# Image Noise Removal

## Requirements:

### Python for displaying images:
- `matplotlib` for images 
- `Pillow` for displaying images

### C++ for gaussian blur code: 
- **OpenMP** support (via `libomp` on macOS)
- MinGW or WSL support for windows 
- C++17 compiler (such as `clang++` or `g++`)
- dependency on stb_image.h and stb_image_write.h to process jpeg in c++
- these are from https://github.com/nothings/stb/tree/master

## Compiling executable for blur: 
```
g++-14 -std=c++17 -fopenmp gaussian.cpp -o gaussian
```
 
## running
 - need to use .jpg files
 - specficy input folder with your .jpg files and output folder to put the blurred .jpg files

### running with python wrapper to display images:
```
python blur_wrapper.py path/to/input_dir/ path/to/output_dir/
```

### running just images blur c++:
```
./gaussian_speed relative-path-input-images relative-path-output-images
```

