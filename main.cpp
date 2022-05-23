#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <numeric>
#include <typeinfo>
#include "breath_detector.h"

#include <string>
using namespace std;
using namespace cv;
int main() {

    string a= to_string(8);
    cout<<a<<endl;
//    VideoCapture cap(0);


    vector<float> tmp = {1,2,3,4,5,6,7,8};

    Mat frameRGB(tmp);
    cout<<"a:"<<frameRGB.type()<<endl;
//    breath_detector breath;
    dft(frameRGB,frameRGB,CV_HAL_DFT_COMPLEX_OUTPUT);

    cout<<"a:"<<frameRGB<<endl;
    cv::pow(frameRGB,2,frameRGB);
    cout<<"b:"<<frameRGB<<endl;
    vector<Mat> channels;
    split(frameRGB,channels);
    add(channels[0],channels[1],channels[0]);
    cout<<"add:"<<channels[0]<<endl;
    pow(channels[0],0.5,channels[0]);
    cout<<"sqrt:"<<channels[0]<<endl;
//    while(true){
//        cap.read(frameRGB);
//        if (frameRGB.empty()){
//            break;
//        }
//
//        cvtColor(frameRGB,frameGray,COLOR_BGR2GRAY);
//        equalizeHist(frameGray,frameGray);
//        breath.process_frame(frameRGB,frameGray);
//        breath.process_frame();
//    }


    return 0;
}
