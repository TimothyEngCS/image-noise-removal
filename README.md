# image-noise-removal
The operator observed is known as Gaussian Image Filtering/Smoothing/Blurring in where based on the Gaussian distribution it performs a weighted average of nearby pixels and is used to remove Gaussian noise and is a realistic model of defocused lens.
The standard deviation influences how significantly the center pixel's nearby pixels affect the results after computation.

This Gaussian Blur program is compiled and coded in Python
In Python code directory, the syntax in which the code is dependent on is two separate directories for inputting a sample image dataset and outputting the dataset after compilation.
In Gaussian Blur directory, the directories are as followed: input_folder (folder with image file(s)), output_folder (empty folder), and PhotoGaussianVF.py (soft final code of Gaussian Smoothing via Python)

Compilation occurs in terminal line execution with the following syntax (I used Windows Powershell for this demonstration): python PhotoGaussianVF.py input_folder <float data type for sigma input> output_folder

Sigma is defined as the amount of blurring dealt onto the image compiled through the code. This float data type requires significantly more calculations per pixel if the user input is higher. In the GitHub directories/files provided, the sigma used for demonstration purposes is 12.0.
Reference Link: https://www.southampton.ac.uk/~msn/book/new_demo/gaussian/
