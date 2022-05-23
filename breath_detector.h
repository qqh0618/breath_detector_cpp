//
// Created by 86176 on 2022/5/17.
//

#ifndef BREATE_DETCT_BREATH_DETECTOR_H
#define BREATE_DETCT_BREATH_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "opencv.hpp"
#include <vector>
using namespace cv;
using namespace std;
using namespace dnn;
using namespace std;

class breath_detector{
public:
    breath_detector(){;}

//    void process_frame(Mat &frameRGB, Mat &frameGray);
    void process_frame();

private:
    int get_frequency(vector<float> &breath_changes, bool graph=false);
    Mat crop_fragment(Mat &frameRGB,Rect &face);
    vector<vector<int>> Mat2Array(Mat image);
    CascadeClassifier haarClassifier;
    Mat ffreq(int n,double d=1.0 );
    Mat count_abs_dft(Mat dft_frame);
};

#endif //BREATE_DETCT_BREATH_DETECTOR_H
