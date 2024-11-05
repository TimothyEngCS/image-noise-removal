#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <omp.h>
#include <cmath>

namespace fs = std::filesystem;

// Custom clamp function to handle values within a range
template <typename T>
T clamp(const T& value, const T& low, const T& high) {
    return (value < low) ? low : (value > high) ? high : value;
}

// Generate and normalize a 1D Gaussian kernel
std::vector<float> gaussian_kernel_1d(int kernel_sz, float sigma_val) {
    std::vector<float> ker(kernel_sz, 0);
    float sum = 0.0f;
    float norm = 1.0f / (std::sqrt(2.0f * M_PI) * sigma_val);
    float denom = 2.0f * sigma_val * sigma_val;

    for (int i = 0; i < kernel_sz; ++i) {
        float diff = i - (kernel_sz / 2);
        ker[i] = norm * std::exp(-(diff * diff) / denom);
        sum += ker[i];
    }

    // Normalize the kernel
    for (int i = 0; i < kernel_sz; ++i) {
        ker[i] /= sum;
    }
    return ker;
}

// Applying Gaussian blur
void apply_gaussian_blur(unsigned char* image_data, int width, int height, int channels, const std::vector<float>& kernel_x, const std::vector<float>& kernel_y) {
    int kernel_sz = kernel_x.size();
    int padding_sz = kernel_sz / 2;

    // Temporary storage for the X and Y blurs
    std::vector<unsigned char> temp_image(width * height * channels, 0);

    // X-direction blur
    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float pixel_accum = 0.0f;
                for (int kx = -padding_sz; kx <= padding_sz; ++kx) {
                    int img_x = clamp(x + kx, 0, width - 1);
                    pixel_accum += kernel_x[kx + padding_sz] * image_data[(y * width + img_x) * channels + c];
                }
                temp_image[(y * width + x) * channels + c] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, pixel_accum)));
            }
        }
    }

    // Y-direction blur
    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float pixel_accum = 0.0f;
                for (int ky = -padding_sz; ky <= padding_sz; ++ky) {
                    int img_y = clamp(y + ky, 0, height - 1);
                    pixel_accum += kernel_y[ky + padding_sz] * temp_image[(img_y * width + x) * channels + c];
                }
                image_data[(y * width + x) * channels + c] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, pixel_accum)));
            }
        }
    }
}

// Function to process and save images with Gaussian blur
void process_image(const std::string& img_path, const std::string& output_path, const std::vector<float>& kernel_x, const std::vector<float>& kernel_y) {
    int width, height, channels;
    unsigned char* image_data = stbi_load(img_path.c_str(), &width, &height, &channels, 0);

    if (!image_data) {
        std::cerr << "Error: Could not load image file " << img_path << std::endl;
        return;
    }

    // Apply Gaussian blur
    apply_gaussian_blur(image_data, width, height, channels, kernel_x, kernel_y);

    // Save the processed image
    stbi_write_jpg(output_path.c_str(), width, height, channels, image_data, 100);
    std::cout << "Processed and saved: " << output_path << std::endl;

    // Free image memory
    stbi_image_free(image_data);
}

int main(int argc, char** argv) {
    std::string input_dir = "images/test/input/"; // Input folder
    std::string output_dir = "images/test/output/"; // Output folder

    // Default values for kernel size and sigma
    int kernel_sz = 7; // Ensure odd kernel size for center point
    float sigma_val = 12.0;

    // Check if input and output folder paths are provided
    if (argc >= 3) {
        input_dir = argv[1];
        output_dir = argv[2];
    }

    // Check if kernel size and sigma are provided as additional arguments
    if (argc >= 5) {
        kernel_sz = std::atoi(argv[3]);
        sigma_val = std::atof(argv[4]);
    }

    // Generate 1D Gaussian kernels for X and Y directions
    std::vector<float> kernel_x = gaussian_kernel_1d(kernel_sz, sigma_val);
    std::vector<float> kernel_y = kernel_x; // Since Gaussian kernel is symmetric

    // Collect all image file paths
    std::vector<std::string> img_paths;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.is_regular_file()) {
            std::string file_ext = entry.path().extension().string();
            if (file_ext == ".jpg") { // Only process jpg files
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

        process_image(img_path, output_path, kernel_x, kernel_y);

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