//
//  tools.hpp
//  Detection
//
//  Created by TonyMaster on 2017/9/21.
//  Copyright © 2017年 Tony_LB. All rights reserved.
//

#ifndef tools_hpp
#define tools_hpp

#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.hpp>
#include <armadillo>


#include <math.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
using namespace cv;

double pi=3.1415926;

#define varName(x) #x
#define printExp(exp) cout<<#exp<<"为:\t\t"<<(exp)<<endl
#define printExpToString(exp) cout<<(string(#exp)+"为:\t\t")<<(exp).toString()<<endl //注意exp加括号更安全

//判断两个Mat是否相等
bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2) {
        int Rows = mat1.rows;
        int Cols = mat1.cols * mat1.channels();
        int IsStopOutLoop = 0;
        bool bRet = true;
        do
        {
            for (int i = 0; i < Rows; i++)
            {
                const uchar *data1 = mat1.ptr<uchar>(i);
                const uchar *data2 = mat2.ptr<uchar>(i);
                for (int j = 0; j < Cols; j++)
                {
                    if (data1[j] != data2[j])
                    {
                        IsStopOutLoop++;
                        bRet = false;
                        break;
                    }
                }
                if (IsStopOutLoop != 0)
                    break;
            }
            //bRet = true;
        } while (false);
        if (bRet == false)
        {
            //如果两幅图片不相等  进行相应的处理  这里就用cout模拟了
            cout << "这两幅Mat是不同的" << endl;
        }
        else
        {
            //如果两幅图片相等   进行相应的处理  这里就用cout模拟了
            cout << "这两幅Mat图片是相同的" << endl;
        }
        return 0;
}

//matlab strel('disk',Radius)
Mat strelDisk(int Radius)
{
    int borderWidth; Mat sel; int m, n;
    switch (Radius){
        case 1:
        case 2:
            if (Radius == 1)
                borderWidth = 1;
            else
                borderWidth = 2;
            sel=Mat((2 * Radius + 1), (2 * Radius + 1), CV_8U, cv::Scalar(1));
            break;//当半径为1时是3X3的 ,当半径为2时是5X5的
        case 3:
            borderWidth = 0;
            sel=Mat((2 * Radius - 1), (2 * Radius - 1), CV_8U, cv::Scalar(1));
            break;
        default:
            n = Radius / 7; m = Radius % 7;
            if (m == 0 || m >= 4)
                borderWidth = 2 * (2 * n + 1);
            else
                borderWidth = 2 * 2 * n;
            sel=Mat((2 * Radius - 1), (2 * Radius - 1), CV_8U, cv::Scalar(1));
            break;
    }
    for (int i = 0; i < borderWidth; i++){
        for (int j = 0; j < borderWidth; j++){
            if (i + j < borderWidth){
                sel.at<uchar>(i, j) = 0;
                sel.at<uchar>(i, sel.cols - 1 - j) = 0;
                sel.at<uchar>(sel.rows - 1 - i, j) = 0;
                sel.at<uchar>(sel.rows - 1 - i, sel.cols - 1 - j) = 0;
            }
        }
    }
    return sel;
}

//int rgb2hsi(Mat &image,Mat &hsi){
//    if(!image.data){
//        cout<<"Miss Data"<<endl;
//        return -1;
//    }
//    int nl = image.rows;
//    int nc = image.cols;
//    if(image.isContinuous()){
//        nc = nc*nl;
//        nl = 1;
//    }
//    for(int i = 0;i < nl;i++){
//        uchar *src = image.ptr<uchar>(i);
//        uchar *dst = hsi.ptr<uchar>(i);
//        for(int j = 0;j < nc;j++){
//            float b = src[j*3]/255.0;
//            float g = src[j*3+1]/255.0;
//            float r = src[j*3+2]/255.0;
//            float num = (float)(0.5*((r-g)+(r-b)));
//            float den = (float)sqrt((r-g)*(r-g)+(r-b)*(g-b));
//            float H,S,I;
//            if(den == 0){	//分母不能为0
//                H = 0;
//            }
//            else{
//                double theta = acos(num/den);
//                if(b <= g)
//                    H = theta/(pi*2);
//                else
//                    H = (2*pi - theta)/(2*pi);
//            }
//            double minRGB = min(min(r,g),b);
//            den = r+g+b;
//            if(den == 0)	//分母不能为0
//                S = 0;
//            else
//                S = 1 - 3*minRGB/den;
//            I = den/3.0;
//            //将S分量和H分量都扩充到[0,255]区间以便于显示;
//            //一般H分量在[0,2pi]之间，S在[0,1]之间
//            dst[3*j] = H*255;
//            dst[3*j+1] = S*255;
//            dst[3*j+2] = I*255;
//        }
//    }
//    return 0;
//}

Mat preProcess(Mat img)
{
//    首先提取出图像的mask，利用红色信道
    vector<Mat> BGRchannels;
    split(img, BGRchannels);
    Mat green = BGRchannels[1];
    Mat red = BGRchannels[2];
    Mat mask = Mat::zeros(red.rows, red.cols, CV_8UC1);
    threshold(red, mask, 15, 1, 0);
    
//    进行掩模操作
    Mat imgRoi = Mat::zeros(red.rows, red.cols, CV_8UC(3));
//    for(int i = 0; i < red.rows; i++)
//        for(int j = 0; j < red.cols; j++)
//            if(mask[i][j] == 255)
    img.copyTo(imgRoi, mask);
    
//    通过参考图片的亮度通道平均值来对所有的图像调整亮度
    double avgL = 70;
    Mat imgRoiLab(red.rows, red.cols, CV_8UC(3));
    cvtColor(imgRoi, imgRoiLab, CV_BGR2Lab);
    vector<Mat> LabChannels;
    split(imgRoiLab, LabChannels);
    Mat L = LabChannels[0];
    double imgAvgL = sum(L).val[0] / sum(mask).val[0];
    //如果小于平均亮度则对其提升，否则不用处理
    if(imgAvgL < avgL)
    {
        L += avgL - imgAvgL;
    }
    merge(LabChannels, imgRoiLab);
    imgRoiLab.copyTo(imgRoiLab, mask);
    cvtColor(imgRoiLab, imgRoi, CV_Lab2BGR);
    imgRoi.copyTo(imgRoi, mask);
    
    
//    中值滤波 3*3
    medianBlur(imgRoi, imgRoi, 3);
    imshow("medBlured", imgRoi);
    
//    对比度增强，多尺度低帽变换
    BGRchannels.clear();
    split(imgRoi, BGRchannels);
    Mat Iwr = Mat::zeros(red.rows, red.cols, CV_8UC(3));
    Mat Iwd = Mat::zeros(red.rows, red.cols, CV_8UC(3));
    Mat Ibr = Mat::zeros(red.rows, red.cols, CV_8UC(3));
    Mat Ibd = Mat::zeros(red.rows, red.cols, CV_8UC(3));
    vector<Mat> Iwrchannels;
    vector<Mat> Iwdchannels;
    vector<Mat> Ibrchannels;
    vector<Mat> Ibdchannels;
    split(Iwr, Iwrchannels);
    split(Iwd, Iwdchannels);
    split(Ibr, Ibrchannels);
    split(Ibd, Ibdchannels);
    Mat filtedImgR = BGRchannels[2];
    Mat filtedImgB = BGRchannels[0];
    Mat filtedImgG = BGRchannels[1];
    for(int i = 2; i < 12; i++)
    {
        //变换kernel
        Mat B = strelDisk(i);
        Mat BPlus = strelDisk(i + 1);
        Mat temp1 = Mat::zeros(red.rows, red.cols, CV_8UC1);
        Mat temp2 = Mat::zeros(red.rows, red.cols, CV_8UC1);
        
        morphologyEx(filtedImgR, temp1, MORPH_TOPHAT, B);
        cv::max(Iwrchannels[2], temp1, Iwrchannels[2]);
        morphologyEx(filtedImgG, temp1, MORPH_TOPHAT, B);
        cv::max(Iwrchannels[1], temp1, Iwrchannels[1]);
        morphologyEx(filtedImgB, temp1, MORPH_TOPHAT, B);
        cv::max(Iwrchannels[0], temp1, Iwrchannels[0]);
        
        morphologyEx(filtedImgR, temp1, MORPH_OPEN, B);
        morphologyEx(filtedImgR, temp2, MORPH_OPEN, BPlus);
        cv::max(Iwdchannels[2], temp1 - temp2, Iwdchannels[2]);
        morphologyEx(filtedImgG, temp1, MORPH_OPEN, B);
        morphologyEx(filtedImgG, temp2, MORPH_OPEN, BPlus);
        cv::max(Iwdchannels[1], temp1 - temp2, Iwdchannels[1]);
        morphologyEx(filtedImgB, temp1, MORPH_OPEN, B);
        morphologyEx(filtedImgB, temp2, MORPH_OPEN, BPlus);
        cv::max(Iwdchannels[0], temp1 - temp2, Iwdchannels[0]);
        
        morphologyEx(filtedImgR, temp1, MORPH_BLACKHAT, B);
        cv::max(Ibrchannels[2], temp1, Ibrchannels[2]);
        morphologyEx(filtedImgG, temp1, MORPH_BLACKHAT, B);
        cv::max(Ibrchannels[1], temp1, Ibrchannels[1]);
        morphologyEx(filtedImgB, temp1, MORPH_BLACKHAT, B);
        cv::max(Ibrchannels[0], temp1, Ibrchannels[0]);
        
        morphologyEx(filtedImgR, temp1, MORPH_CLOSE, B);
        morphologyEx(filtedImgR, temp2, MORPH_CLOSE, BPlus);
        cv::max(Ibdchannels[2], temp2 - temp1, Ibdchannels[2]);
        morphologyEx(filtedImgG, temp1, MORPH_CLOSE, B);
        morphologyEx(filtedImgG, temp2, MORPH_CLOSE, BPlus);
        cv::max(Ibdchannels[1], temp2 - temp1, Ibdchannels[1]);
        morphologyEx(filtedImgB, temp1, MORPH_CLOSE, B);
        morphologyEx(filtedImgB, temp2, MORPH_CLOSE, BPlus);
        cv::max(Ibdchannels[0], temp2 - temp1, Ibdchannels[0]);
    }
    //忘了合并信道了...
    merge(Iwrchannels, Iwr);
    merge(Iwdchannels, Iwd);
    merge(Ibrchannels, Ibr);
    merge(Ibdchannels, Ibd);
    
    
    
//    vector<Mat> HSIchannels; 利用Hsi颜色空间，
//    split(imgRoiHsi, HSIchannels);
//    Mat i = HSIchannels[2];
//    double srcAvgI = 0.4;
//    double avgI = sum(i).val[0] / sum(mask).val[0];
//    //亮度没有达到平均值的时候
//    if(avgI < srcAvgI)
//    {
//        double Idiff = srcAvgI - avgI;
//        
//    }
    
    
//    Mat testHsi(red.rows, red.cols, CV_8UC(3)); 测试两个RGB2HSI结果是否相等
//    vector<Mat> HSIchannels;
//    HSIchannels.push_back(h);
//    HSIchannels.push_back(s);
//    HSIchannels.push_back(i);
//    merge(HSIchannels, testHsi);
//    if(matIsEqual(imgRoiHsi, testHsi))
//        cout<< "RRRR" << endl;
//    else
//        cout<< "NNNN" << endl;
    
    
    return imgRoi + Iwr + Iwd - Ibr - Ibd;
}

#endif /* tools_hpp */
