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
    VideoCapture cap(0);

    Mat frameRGB, frameGray;
    breath_detector breath;
//    while(true){
//        cap.read(frameRGB);
//        if (frameRGB.empty()){
//            break;
//        }
//
//        cvtColor(frameRGB,frameGray,COLOR_BGR2GRAY);
//        equalizeHist(frameGray,frameGray);
//        breath.process_frame(frameRGB,frameGray);
        breath.process_frame();
//    }


    return 0;
}
