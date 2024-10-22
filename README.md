# image-noise-removal

- python implementation requires mpi4py from MPI
- c++ implementation requires installing OpenCV (we used it for displaying images )

compiling with the c++ version:
-- for linux/macos: clang++ -fopenmp -std=c++17 gaussian_speed.cpp -I/usr/local/Cellar/opencv/4.10.0_11/include/opencv4 -L/usr/local/Cellar/opencv/4.10.0_11/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -o gaussian_speed

-- for windows: g++ -fopenmp -std=c++17 gaussian_speed.cpp -IC:/path_to_opencv/build/include -LC:/path_to_opencv/build/x64/mingw/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -o gaussian_speed.exe

