#include <opencv2/opencv.hpp>  //include opencv lib
#include <iostream> //for io operations
#include <filesystem> // for directory traversal (C++17)
#include <omp.h> // Include OpenMP
namespace fs = std::filesystem;

// struct helps to display images and is used as the data for the mouse callback when clicking image left/right side
struct CallbackData {
    const std::vector<std::pair<cv::Mat, cv::Mat>>& image_pairs;
    int* current_index;
};

//function to make gaussian kernel -- creating custom filter
std::vector<std::vector<float>> gaussian_kernel(int kernel_sz, float sigma_val) {
    std::vector<std::vector<float>> ker(kernel_sz, std::vector<float>(kernel_sz, 0)); //initializing kernel
    int cntr = kernel_sz / 2; //calculating center point of kernel
    float s = 0.0; //sum for normalizing later
    
    //populating kernel with gaussian values
    for (int i = 0; i < kernel_sz; ++i) {
        for (int j = 0; j < kernel_sz; ++j) {
            float diffx = i - cntr; //difference in x-direction
            float diffy = j - cntr; //difference in y-direction
            
            // calculating kernel values using gaussian formula
            ker[i][j] = (1.0 / (2 * M_PI * sigma_val * sigma_val)) * std::exp(-(diffx * diffx + diffy * diffy) / (2 * sigma_val * sigma_val)); 
            s += ker[i][j]; // add to sum for normalizing
        }
    }
    
    //normalize kernel so the sum is 1
    for (int i = 0; i < kernel_sz; ++i) {
        for (int j = 0; j < kernel_sz; ++j) {
            ker[i][j] /= s; // divide by sum
        }
    }

    return ker; //return the kernel matrix
}

//applying gaussian filter to image using convolution -- this does the blur
cv::Mat apply_gaussian_blur(const cv::Mat& img, const std::vector<std::vector<float>>& ker) {
    int kernel_sz = ker.size(); // size of kernel
    int padding_sz = kernel_sz / 2; //size of padding for edges
    int img_h = img.rows; //height of image
    int img_w = img.cols; //width of image
    int ch = img.channels(); // number of color channels (rgb)
    
    //padding the image to handle edges
    cv::Mat padded_img;
    cv::copyMakeBorder(img, padded_img, padding_sz, padding_sz, padding_sz, padding_sz, cv::BORDER_CONSTANT, cv::Scalar(0)); 

    cv::Mat blurred_img = cv::Mat::zeros(img.size(), CV_32FC3); //initialize result image with zeros

    //perform convolution over image
    for (int i = 0; i < img_h; ++i) {
        for (int j = 0; j < img_w; ++j) {
            for (int c = 0; c < ch; ++c) {  //iterate over rgb channels
                float pixel_accum = 0.0; // accumulator for the pixel value
                
                //apply kernel to surrounding pixels
                for (int ki = 0; ki < kernel_sz; ++ki) {
                    for (int kj = 0; kj < kernel_sz; ++kj) {
                        int ni = i + ki; //new i index in padded image
                        int nj = j + kj; //new j index in padded image
                        
                        //sum up weighted values
                        pixel_accum += padded_img.at<cv::Vec3b>(ni, nj)[c] * ker[ki][kj];
                    }
                }
                blurred_img.at<cv::Vec3f>(i, j)[c] = pixel_accum; //save result in the blurred image
            }
        }
    }

    return blurred_img; //return blurred image
}


//handle the clicking between the photos. treat right click as next left as previous
// Mouse callback function to handle click events
void onMouse(int event, int x, int y, int, void* userdata) {
    // Cast userdata back to the correct type
    CallbackData* data = static_cast<CallbackData*>(userdata);
    int img_width = data->image_pairs[*(data->current_index)].first.cols;

    // Check for left button click
    if (event == cv::EVENT_LBUTTONDOWN) {
        if (x < img_width / 2) {
            // Clicked on the left side (previous image)
            *(data->current_index) = (*(data->current_index) - 1 + data->image_pairs.size()) % data->image_pairs.size();
        } else {
            // Clicked on the right side (next image)
            *(data->current_index) = (*(data->current_index) + 1) % data->image_pairs.size();
        }
    }
}

// Function to display input and output images side by side with click navigation
void display_images_side_by_side(const std::vector<std::pair<cv::Mat, cv::Mat>>& image_pairs) {
    int current_index = 0;

    // Structure to hold callback data
    CallbackData data = { image_pairs, &current_index };

    // Set up the mouse callback function
    cv::namedWindow("Input vs Output");
    cv::setMouseCallback("Input vs Output", onMouse, &data);

    while (true) {
        // Get the current input-output image pair
        cv::Mat input_img = image_pairs[current_index].first;
        cv::Mat output_img = image_pairs[current_index].second;

        // Concatenate input and output images side by side
        cv::Mat concatenated;
        cv::hconcat(input_img, output_img, concatenated);

        // Show the concatenated image
        cv::imshow("Input vs Output", concatenated);

        // Wait for a key press
        int key = cv::waitKey(30);
        // Check for space bar (32) or ESC key (27) to exit
        if (key == 32 || key == 27) {
            break;
        }
    }

    cv::destroyAllWindows();  // Close all OpenCV windows
}

int main() {
    //define input and output directories - potentially better to be arguments to running the program. 
    std::string input_dir = "/path/to/input/folder/";   // Update this path to your input folder
    std::string output_dir = "/path/to/output/folder/"; // Update this path to your output folder
    input_dir = "images/test/input/";
    output_dir = "images/test/output/";

    //define kernel size and standard deviation -- affects blur strength
    int kernel_sz = 5;  //larger value for more blur
    float sigma_val = 12.0;    //higher sigma means stronger blur effect

    //generate gaussian kernel
    std::vector<std::vector<float>> kernel = gaussian_kernel(kernel_sz, sigma_val);

    // vector for the before and after images
    std::vector<std::pair<cv::Mat, cv::Mat>> image_pairs;

    // Collect all image file paths and their filenames for OpenMP parallelization
    std::vector<std::string> img_paths;
    std::vector<std::string> img_filenames;
    for (const auto& entry : std::filesystem::directory_iterator(input_dir)) {
        if (entry.is_regular_file()) {
            img_paths.push_back(entry.path().string());
            img_filenames.push_back(entry.path().filename().string());  // Collect filenames once
        }
    }

    // Process images in parallel using OpenMP
    #pragma omp parallel for shared(image_pairs)
    for (int i = 0; i < img_paths.size(); ++i) {
        std::cout << "Processing: " << img_paths[i] << std::endl;

        // Load the image
        cv::Mat img = cv::imread(img_paths[i]);
        if (img.empty()) {
            std::cerr << "Error: Could not open image file " << img_paths[i] << std::endl;
            continue;
        }

        // Apply Gaussian blur
        cv::Mat blurred_img = apply_gaussian_blur(img, kernel);
        cv::Mat blurred_img_8u;
        blurred_img.convertTo(blurred_img_8u, CV_8UC3); // Convert to 8-bit unsigned format

        // Save the blurred image to the output folder
        std::string output_path = output_dir + img_filenames[i];  // Use pre-collected filename
        cv::imwrite(output_path, blurred_img_8u);
        std::cout << "Saved to: " << output_path << std::endl;

        // Add to image_pairs vector (critical section for thread safety)
        #pragma omp critical
        {
            image_pairs.push_back({img, blurred_img_8u});
        }
    }

    // If no images were processed, exit
    if (image_pairs.empty()) {
        std::cerr << "No images processed." << std::endl;
        return -1;
    }
    // Display the images side by side and allow navigation
    display_images_side_by_side(image_pairs);

    return 0; //exit program
}

//install open:cv
///compile with clang++ -std=c++17 gaussian_speed.cpp -I/usr/local/Cellar/opencv/4.10.0_11/include/opencv4 -L/usr/local/Cellar/opencv/4.10.0_11/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -o gaussian_speed
