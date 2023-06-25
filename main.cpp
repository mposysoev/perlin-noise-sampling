#include "FastNoiseLite.h"
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <random>
#include <string>

static void help()
{
    std::cout << "USAGE: ./perlin-noise-texture <name of input file> <percent of random sampling> <frequency for OpenSimplex2S noise>" << std::endl;
}

void show_image(std::string name_of_window, cv::Mat image)
{
    // Create a window to display the image
    cv::namedWindow(name_of_window, cv::WINDOW_NORMAL);

    // Display the image in the window
    cv::imshow(name_of_window, image);
}

void monte_carlo_sampling(cv::Mat image, int percent, std::string input_name)
{
    cv::Mat monteCarloRandomSampledImage = image.clone();
    monteCarloRandomSampledImage.setTo(cv::Scalar(0, 0, 0));

    int width = image.cols;
    int height = image.rows;

    int number_of_iter = width * height * percent / 100;
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<int> distX(0, width - 1);
    std::uniform_int_distribution<int> distY(0, height - 1);

    for (int i = 0; i < number_of_iter; i++) {
        int random_pixel_X = distX(gen);
        int random_pixel_Y = distY(gen);

        cv::Vec3b pixel = image.at<cv::Vec3b>(random_pixel_Y, random_pixel_X);

        monteCarloRandomSampledImage.at<cv::Vec3b>(random_pixel_Y, random_pixel_X) = pixel;
    }

    show_image("original", image);
    show_image("monteCarloSampled", monteCarloRandomSampledImage);

    std::string param = std::to_string(percent);
    std::string output_name = "monte-carlo-" + param + input_name;

    cv::imwrite(output_name, monteCarloRandomSampledImage);
}

void perlin_mask_sampling(cv::Mat image, float freq, std::string input_name)
{
    // Create a FastNoise object for Perlin noise generation
    FastNoiseLite noise;

    int width = image.cols;
    int height = image.rows;

    std::random_device rd;
    std::mt19937 gen(rd());

    // Set the noise parameters
    noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2S);
    noise.SetFrequency(freq); // Adjust the frequency for different levels of detail

    // Generate the Perlin noise mask
    cv::Mat perlinMask(height, width, CV_8UC1);
    for (int y = 0; y < height - 1; y++) {
        for (int x = 0; x < width - 1; x++) {
            float noiseValue = noise.GetNoise((float)x, (float)y);
            perlinMask.at<uchar>(y, x) = static_cast<uchar>((noiseValue + 1.0f) * 0.5f * 255.0f);
        }
    }

    // Apply the Perlin noise mask to the image
    cv::Mat perlinImage = image.clone();
    perlinImage.setTo(cv::Scalar(0, 0, 0));

    std::uniform_int_distribution<int> distProbability(0, 255);
    for (int y = 0; y < height - 1; y++) {
        for (int x = 0; x < width - 1; x++) {
            int noiseValue = perlinMask.at<uchar>(y, x);
            if (noiseValue > distProbability(gen)) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
                perlinImage.at<cv::Vec3b>(y, x) = pixel;
            }
        }
    }

    show_image("Perlin Mask", perlinMask);
    show_image("Perlin Image", perlinImage);

    std::string param = std::to_string(freq);
    std::string output_name = "perlin-noise-" + param + input_name;

    cv::imwrite(output_name, perlinImage);
}

int main(int argc, char* argv[])
{
    if (argc != 4) {
        help();
        return -1;
    }

    auto file_path = argv[1];
    auto percent = std::stoi(argv[2]);
    auto frequence = std::stof(argv[3]);

    // Load an image from file
    cv::Mat image = cv::imread(file_path);

    if (image.empty()) {
        std::cout << "Failed to load image." << std::endl;
        return -1;
    }

    show_image("original", image);

    // Sampling random pixels from image
    monte_carlo_sampling(image, percent, file_path);

    // Sampling of all pixels via Perlin noise mask
    perlin_mask_sampling(image, frequence, file_path);

    // Wait for a key press
    cv::waitKey(0);

    // Close the window
    cv::destroyAllWindows();

    return 0;
}
