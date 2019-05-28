/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
/* This sample queries the properties of the CUDA devices present in the system via CUDA Runtime API. */

// Shared Utilities (QA Testing)

// std::system includes
#include <memory>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching.hpp>
// #include <opencv2/core/utility.hpp>
#include <opencv2/core/utility.hpp>

// CUDA-C includes
#include <cuda.h>

// This function wraps the CUDA Driver API into a template function
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

using namespace std;
using namespace cv;

bool try_use_gpu = false;
bool divide_images = false;
cv::Stitcher::Mode mode = cv::Stitcher::PANORAMA;
std::vector<cv::Mat> imgs;
std::string result_name = "result.jpg";

void printUsage(char** argv);
int parseCmdArgs(int argc, char** argv);


int main(int argc, char **argv)
{
    std::cout<<"Starting..."<<argv[0]<<std::endl<<std::endl;
    std::cout<<"CUDA Device Query (Runtime API) version (CUDART static linking)"<<std::endl<<std::endl;
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    
    if (error_id != cudaSuccess){
        std::cout<<"cudaGetDeviceCount returned "<< (int)error_id<< cudaGetErrorString(error_id)<<std::endl;
        std::cout<<"Result = FAIL"<<std::endl;
        exit(EXIT_FAILURE);
    }
    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
        std::cout<<"There are no available device(s) that support CUDA"<<std::endl;
    else
        std::cout<<"Detected "<< deviceCount<<" CUDA Capable device(s)"<<std::endl<<std::endl;
    int dev, driverVersion = 0, runtimeVersion = 0;
    for (dev = 0; dev < deviceCount; ++dev){
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        std::cout<<"Device: "<<dev<<" "<<deviceProp.name<<std::endl;
        // Console log
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        std::cout<<"CUDA Driver Version / Runtime Version          "<< driverVersion/1000
                << (driverVersion%100)/10<<runtimeVersion/1000<<(runtimeVersion%100)/10<<std::endl;
        std::cout<<"CUDA Capability Major / Minor version number:          "<< driverVersion/1000
                << deviceProp.major<<deviceProp.minor<<std::endl;
    }
    // Check OpenCV Version 
    std::cout << "OpenCV version : " << CV_VERSION << std::endl;
    std::cout << "Major version : " << CV_MAJOR_VERSION << std::endl;
    std::cout << "Minor version : " << CV_MINOR_VERSION << std::endl;
    std::cout << "Subminor version : " << CV_SUBMINOR_VERSION << std::endl;

    cv::Mat image;
    cv::Mat E = cv::Mat::eye(4,4,CV_64F);
    std:: cout<<"E = "<< std::endl<<" "<<E<<std::endl<<std::endl;
    image = cv::imread("images/HappyFish.jpg");
    if (!image.data){
        std::cout<<"Could not open or find the image"<<std::endl;
        return -1;
    }
    std::cout<<"Image open!"<<std::endl;
    cv::Mat imgGrayScale;
    cv::cvtColor(image, imgGrayScale,cv::COLOR_BGR2GRAY);    
    cv::imwrite("grayScale.png",imgGrayScale);
    std::cout<<"Writed Image"<<std::endl;

    for (int i = 1; i < argc; ++i) 
    { 
            // Read the ith argument or image  
            // and push into the image array 
            Mat img = imread(argv[i]); 
            if (img.empty()) 
            { 
                // Exit if image is not present 
                cout << "Can't read image '" << argv[i] << "'\n"; 
                return -1; 
            } 
            imgs.push_back(img); 
    } 
      
    // Define object to store the stitched image 
    Mat pano; 
    // Create a Stitcher class object with mode panoroma 
    Ptr<Stitcher> stitcher = Stitcher::create(mode, false); 
    // Command to stitch all the images present in the image array 
    Stitcher::Status status = stitcher->stitch(imgs, pano); 
    if (status != Stitcher::OK) 
    { 
        // Check if images could not be stiched 
        // status is OK if images are stiched successfully 
        cout << "Can't stitch images\n"; 
        return -1; 
    } 
    // Store a new image stiched from the given  
    //set of images as "result.jpg" 
    imwrite("result.jpg", pano); 
    return 0;
}

/*
void printUsage(char** argv){
    cout <<
         "Images stitcher.\n\n" << "Usage :\n" << argv[0] <<" [Flags] img1 img2 [...imgN]\n\n"
         "Flags:\n"
         "  --d3\n"
         "      internally creates three chunks of each image to increase stitching success\n"
         "  --try_use_gpu (yes|no)\n"
         "      Try to use GPU. The default value is 'no'. All default values\n"
         "      are for CPU mode.\n"
         "  --mode (panorama|scans)\n"
         "      Determines configuration of stitcher. The default is 'panorama',\n"
         "      mode suitable for creating photo panoramas. Option 'scans' is suitable\n"
         "      for stitching materials under affine transformation, such as scans.\n"
         "  --output <result_img>\n"
         "      The default is 'result.jpg'.\n\n"
         "Example usage :\n" << argv[0] << " --d3 --try_use_gpu yes --mode scans img1.jpg img2.jpg\n";
}
int parseCmdArgs(int argc, char** argv){
    if (argc == 1)
    {
        printUsage(argv);
        return EXIT_FAILURE;
    }
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
        {
            printUsage(argv);
            return EXIT_FAILURE;
        }
        else if (string(argv[i]) == "--try_use_gpu")
        {
            if (string(argv[i + 1]) == "no")
                try_use_gpu = false;
            else if (string(argv[i + 1]) == "yes")
                try_use_gpu = true;
            else
            {
                cout << "Bad --try_use_gpu flag value\n";
                return EXIT_FAILURE;
            }
            i++;
        }
        else if (string(argv[i]) == "--d3")
        {
            divide_images = true;
        }
        else if (string(argv[i]) == "--output")
        {
            result_name = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--mode")
        {
            if (string(argv[i + 1]) == "panorama")
                mode = Stitcher::PANORAMA;
            else if (string(argv[i + 1]) == "scans")
                mode = Stitcher::SCANS;
            else
            {
                cout << "Bad --mode flag value\n";
                return EXIT_FAILURE;
            }
            i++;
        }
        else
        {
            Mat img = imread(cv::samples::findFile(argv[i]));
            if (img.empty())
            {
                cout << "Can't read image '" << argv[i] << "'\n";
                return EXIT_FAILURE;
            }
            if (divide_images)
            {
                Rect rect(0, 0, img.cols / 2, img.rows);
                imgs.push_back(img(rect).clone());
                rect.x = img.cols / 3;
                imgs.push_back(img(rect).clone());
                rect.x = img.cols / 2;
                imgs.push_back(img(rect).clone());
            }
            else
                imgs.push_back(img);
        }
    }
    return EXIT_SUCCESS;
}
*/