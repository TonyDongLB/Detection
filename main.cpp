//
//  main.cpp
//  Detection
//
//  Created by TonyMaster on 2017/9/21.
//  Copyright © 2017年 Tony_LB. All rights reserved.
//

#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.hpp>
#include <armadillo>
#include <Eigen/Dense>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

#include "tools.hpp"
#include "OD_detect.hpp"

using namespace std;
using namespace cv;

int main(int argc, const char * argv[])
{
    string imgPath = "/Users/apple/Desktop/using/images/ddb1_fundusimages/image013.png";
    Mat origin = imread(imgPath);
    
    Mat img = origin.clone();
//    int col = 750;
//    int row = int((750.0 / origin.size().height) * origin.size().width);
//    Mat img(row, col, CV_8UC(3), Scalar::all(0));
//    resize(origin, img, Size(row, col));
    Mat preProcessed(origin.rows, origin.cols, CV_8UC(3));
    preProcessed = preProcess(img);
//    cout<< origin.rows << " " << origin.cols << endl;
    Mat deletedOD = OD_detect(imgPath, preProcessed);
    
    

    
    
    waitKey(0);
}
