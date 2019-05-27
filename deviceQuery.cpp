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
#include <opencv2/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/core/utility.hpp>

// CUDA-C includes
#include <cuda.h>

// This function wraps the CUDA Driver API into a template function
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

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
    cv::cvtColor(imgGrayScale, image, cv::COLOR_BGR2GRAY);    
    cv::imwrite("srcImageGrayScale.png",imgGrayScale);
    return 0;
}
