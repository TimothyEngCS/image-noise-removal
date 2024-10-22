# Image Noise Removal

## Requirements:

### Python Implementation:
- `mpi4py` from MPI
- `Pillow` for displaying images

### C++ Implementation:
- **OpenCV** (we use it for displaying images)
- **OpenMP** support (via `libomp` on macOS)
- C++17 compiler (such as `clang++` or `g++`)

## Compiling the C++ Version:

### For Linux/macOS:
```
clang++ -Xpreprocessor -fopenmp -std=c++17 gaussian_speed.cpp -o gaussian_speed `pkg-config --cflags --libs opencv4` -I/usr/local/opt/libomp/include -L/usr/local/opt/libomp/lib -lomp
```

### For windows:
```
g++ -fopenmp -std=c++17 -Ipath_to_opencv\include -Lpath_to_opencv\x64\mingw\lib gaussian_speed.cpp -o gaussian_speed -lopencv_core452 -lopencv_imgcodecs452 -lopencv_highgui452 -lopencv_imgproc452
```

## running
 - need to use .jpg files
 - specficy input folder with your .jpg files and output folder to put the blurred .jpg files

### python:
```
mpiexec -n <number_of_processes> python gaussian_mpi.py <input_dir> <output_dir>
```

### running c++:
```
./gaussian_speed relative-path-input-images relative-path-output-images
```

