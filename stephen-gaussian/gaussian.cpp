#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"  // For loading images
#include "stb_image_write.h"  // For saving images
#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>  // For timing
#include <omp.h>

namespace fs = std::filesystem;

// Function to generate a Gaussian kernel
std::vector<std::vector<float>> gaussian_kernel(int kernel_sz, float sigma_val) {
    std::vector<std::vector<float>> ker(kernel_sz, std::vector<float>(kernel_sz, 0));
    int cntr = kernel_sz / 2;
    float s = 0.0;

    // Populate kernel with Gaussian values
    for (int i = 0; i < kernel_sz; ++i) {
        for (int j = 0; j < kernel_sz; ++j) {
            float diffx = i - cntr;
            float diffy = j - cntr;
            ker[i][j] = (1.0 / (2 * M_PI * sigma_val * sigma_val)) * std::exp(-(diffx * diffx + diffy * diffy) / (2 * sigma_val * sigma_val));
            s += ker[i][j];
        }
    }

    // Normalize kernel so the sum is 1
    for (int i = 0; i < kernel_sz; ++i) {
        for (int j = 0; j < kernel_sz; ++j) {
            ker[i][j] /= s;
        }
    }

    return ker;
}

// Function to apply Gaussian blur
void apply_gaussian_blur(unsigned char* image_data, int width, int height, int channels, const std::vector<std::vector<float>>& kernel) {
    int kernel_sz = kernel.size();
    int padding_sz = kernel_sz / 2;

    // Prepare a padded image (to handle edges during convolution)
    std::vector<unsigned char> padded_image((width + 2 * padding_sz) * (height + 2 * padding_sz) * channels, 0);

    // Copy original image into the padded area
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                padded_image[((y + padding_sz) * (width + 2 * padding_sz) + (x + padding_sz)) * channels + c] =
                    image_data[(y * width + x) * channels + c];
            }
        }
    }

    // Apply Gaussian blur
    std::vector<unsigned char> blurred_image(width * height * channels, 0);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float pixel_accum = 0.0f;

                // Convolve kernel with surrounding pixels
                for (int ky = 0; ky < kernel_sz; ++ky) {
                    for (int kx = 0; kx < kernel_sz; ++kx) {
                        int img_y = y + ky;
                        int img_x = x + kx;

                        pixel_accum += kernel[ky][kx] *
                            padded_image[((img_y) * (width + 2 * padding_sz) + img_x) * channels + c];
                    }
                }
                blurred_image[(y * width + x) * channels + c] = static_cast<unsigned char>(pixel_accum);
            }
        }
    }

    // Copy the blurred image back to the original image data
    std::copy(blurred_image.begin(), blurred_image.end(), image_data);
}

// Function to process and save images with Gaussian blur
void process_image(const std::string& img_path, const std::string& output_path, const std::vector<std::vector<float>>& kernel) {
    int width, height, channels;
    unsigned char* image_data = stbi_load(img_path.c_str(), &width, &height, &channels, 0);

    if (!image_data) {
        std::cerr << "Error: Could not load image file " << img_path << std::endl;
        return;
    }

    // Apply Gaussian blur
    apply_gaussian_blur(image_data, width, height, channels, kernel);

    // Save the processed image
    stbi_write_jpg(output_path.c_str(), width, height, channels, image_data, 100);
    std::cout << "Processed and saved: " << output_path << std::endl;

    // Free image memory
    stbi_image_free(image_data);
}

int main(int argc, char** argv) {
    std::string input_dir = "images/test/input/";  // Input folder
    std::string output_dir = "images/test/output/"; // Output folder
    
    // Default values for kernel size and sigma
    int kernel_sz = 7;      // Default kernel size
    float sigma_val = 12.0; // Default sigma value

    // Check if input and output folder paths are provided
    if (argc >= 3) { 
        input_dir = argv[1]; 
        output_dir = argv[2]; 
    }

    // Check if kernel size and sigma are provided as additional arguments
    if (argc >= 5) {
        kernel_sz = std::atoi(argv[3]);      // Convert third argument to int for kernel size
        sigma_val = std::atof(argv[4]);      // Convert fourth argument to float for sigma value
    }

    // Generate Gaussian kernel
    std::vector<std::vector<float>> kernel = gaussian_kernel(kernel_sz, sigma_val);

    // Collect all image file paths
    std::vector<std::string> img_paths;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.is_regular_file()) {
            std::string file_ext = entry.path().extension().string();
            if (file_ext == ".jpg") {  // Only process jpg files
                img_paths.push_back(entry.path().string());
            }
        }
    }

    // Start total processing time
    auto total_start = std::chrono::high_resolution_clock::now();

    // Process images in parallel using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < img_paths.size(); ++i) {
        std::string img_path = img_paths[i];
        std::string output_path = output_dir + "/" + fs::path(img_path).filename().string();
        
        // Start timing for the current image
        auto start = std::chrono::high_resolution_clock::now();
        
        process_image(img_path, output_path, kernel);
        
        // Calculate elapsed time for the current image
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        // Print out time taken to process the image
        std::cout << "Time taken for " << img_path << ": " << elapsed.count() << " seconds." << std::endl;
    }

    // Calculate total processing time
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = total_end - total_start;
    
    // Print total processing time
    std::cout << "Total processing time for all images: " << total_elapsed.count() << " seconds." << std::endl;

    std::cout << "Processing complete!" << std::endl;
    return 0;
}
