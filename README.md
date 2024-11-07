# Image Noise Removal

## Requirements for running each group members implementation:
### Python for displaying images:
- `matplotlib` for images 
- `Pillow` for displaying images
- tensorflow
- os-sys
- matplotlib
- pillow
- numpy
- opencv-python -- for michael's implementation
- multiprocess - for Todd's implementation
- scipy
- argparse
- concurrent.futures

### C++ for gaussian blur code: 
- **OpenMP** support (via `libomp` on macOS)
- MinGW or WSL support for windows 
- C++17 compiler (such as `clang++` or `g++`)
- dependency on stb_image.h and stb_image_write.h to process jpeg in c++ these are from https://github.com/nothings/stb/tree/master
 
## running 
 - all images are located in images/input
 - for demoing best to use images/input-test which only has a few images. 
 - images should be .jpg files
 - must include argument -s 4 to specify size fo the kernal
 - must include arugment -g 1.5 to specify the degree of blur 

### running with python wrapper to compare implementations with no displaying:
```
python3 blur-benchmarker.py /images/input-test/ /images/output/ -s 5 -g 1.5
```

### running to just display and demo:
```
python3 blur-benchmarker.py images/input-test/ images/output/ -i <implementation> -s 5 -g 1.5 --display

```
 - implementation can be
 - john-vega-gaussian
 - stephen-gaussian
 - timothy-gaussian
 - todd-gaussian
 - michael-gaussian

