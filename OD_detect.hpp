//
//  OD_detect.hpp
//  Detection
//
//  Created by TonyMaster on 2017/9/23.
//  Copyright © 2017年 Tony_LB. All rights reserved.
//

#ifndef OD_detect_hpp
#define OD_detect_hpp

#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.hpp>
#include <armadillo>
#include <mlpack/core.hpp>

#include <math.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
using namespace cv;
using namespace mlpack;


#define varName(x) #x
#define printExp(exp) cout<<#exp<<"为:\t\t"<<(exp)<<endl
#define printExpToString(exp) cout<<(string(#exp)+"为:\t\t")<<(exp).toString()<<endl //注意exp加括号更安全

Mat OD_detect(string imgPath, Mat fen)
{
    Mat orginImg = imread(imgPath);
    int imgw = orginImg.size().width;
    int imgh = orginImg.size().height;
    Mat ODDetectedIMG = Mat::zeros(imgw, imgh, CV_8UC3);
    
    Mat imageGray = Mat::zeros(imgw, imgh, CV_64F);
    cvtColor(orginImg, imageGray, CV_BGR2GRAY);
    vector<Mat> BGRChannels;
    split(orginImg, BGRChannels);
    Mat testImg_red = Mat::zeros(imgw, imgh, CV_64F);
    BGRChannels[2].copyTo(testImg_red);
    
    resize(imageGray, imageGray, cv::Size(116, 150));
    resize(testImg_red, testImg_red, cv::Size(116, 150));
    imshow(varName(imageGray), imageGray);
    imshow(varName(testImg_red), testImg_red);
    
    arma::mat data;
    
//    GaussianBlur(imageGray, imageGray, cv::Size(5, 5), );
//    GaussianBlur(testImg_red, testImg_red, cv::Size(5, 5), );
    
    


    
    
    
    return ODDetectedIMG;
}

#endif /* OD_detect_hpp */
