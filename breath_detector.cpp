//
// Created by 86176 on 2022/5/17.
//
#include <numeric>
#include "breath_detector.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <vector>
#include "opencv.hpp"
#include <algorithm>
#include <algorithm>
#include <string>
#include <bits/stdc++.h>
using namespace cv;
using namespace dnn;
using namespace std;

const int  DETECTION_CHANGE_PERIOD = 120;
const int FPS = 30;
const string HAAR_CLASSIFIER_PATH = "haarcascade_frontalface_alt.xml";


Mat breath_detector::crop_fragment(Mat &frameRGB, Rect &face) {
    int x = face.x;
    int y = face.y;
    int w = face.width;
    int h = face.height;

    return frameRGB(Range(y+floor(h/15), (y+floor(h/5))),Range(x+floor(w/3),x+2*floor(w/3)));
}
double my_round(double r){
    return (r>0.0)?floor(r+0.5):ceil(r-0.5);
}
vector<vector<int>> breath_detector::Mat2Array(Mat image){
    //得宽高
    int w = image.cols;
    int h = image.rows;
    vector<vector<int>> image_arr;
    image_arr.resize(h);
    // 得到初始位置的迭代器
    Mat_<uchar>::iterator it = image.begin<uchar>();

    //得到终止位置的迭代器
    Mat_<uchar>::iterator itend = image.end<uchar>();
    for (size_t i=0;i<h;i++){
        for (size_t j=0;j<w;j++){
            int pixel = *(it+i*w+j);
            image_arr[i].push_back(pixel);
        }
    }
    return image_arr;
}

Mat breath_detector::ffreq(int n, double d) {
    double val =1.0/(n*d);
    int N = floor((n-1)/2)+1;
    vector<float> vec;
    vector<float> vec1(N);
    vector<float> vec2(n-N);
    iota(vec1.begin(),vec1.end(),0);
    iota(vec2.begin(),vec2.end(),-floor(n/2));
    vec.insert(vec.end(),vec1.begin(),vec1.end());
    vec.insert(vec.end(),vec2.begin(),vec2.end());
    Mat vec_frame(vec);
    cout<<"test6"<<endl;
    Mat tmp_frame(vec_frame.rows,vec_frame.cols,CV_32FC1,Scalar(1.0));
    cout<<"test7"<<endl;
    int type_vec = vec_frame.type();
    int typy_tmp = tmp_frame.type();
//    int type_frame = vec_frame.type();
    multiply(vec_frame,tmp_frame,vec_frame,val);//vec_frame 与 tmp_frame逐元素相乘，即vec_frame变为原来的val倍
    cout<<"test8"<<endl;
    return vec_frame;
}

int breath_detector::get_frequency(vector<float> &breath_changes,bool graph){
    vector<float> one_chunk = breath_changes;
    float one_chunk_mean = std::accumulate(one_chunk.begin(),one_chunk.end(),0)/one_chunk.size();
    Mat chunk_frame(one_chunk);
    cout<<"test1"<<endl;
    Mat mean_frame(chunk_frame.rows,chunk_frame.cols,CV_32FC1,Scalar(-one_chunk_mean));
    cout<<"test2"<<endl;
    cv::add(chunk_frame,mean_frame,mean_frame);
    cv::dft(mean_frame,mean_frame,DFT_COMPLEX_OUTPUT);
    Mat ffreq_frame = ffreq(one_chunk.size(),1.0/FPS);
    cout<<"test3"<<endl;
    Mat mask_frame;
    compare(ffreq_frame,0.28,mask_frame,CMP_GE);//比较大小，大于0.28的输出为255，否则输出为0,留下这个掩膜
    //掩码用按位与，无论正负，非0数按位与后均为正数
    Mat abs_dft(mean_frame);//计算傅里叶变换后 复数的模
    cout<<"test4"<<endl;
//    Mat mask_dft;//存放掩码后的数据
    bitwise_and(abs_dft,mask_frame,abs_dft);//abs_dft掩码后的数据
    cout<<"test5"<<endl;
    //找掩码后最大值的位置
//    int maxVaule = *max_element(((vector<double>)abs_dft).begin(),((vector<double>)abs_dft).end());
    int maxPosition = max_element(((vector<double>)abs_dft.reshape(1,1)).begin(),((vector<double>)abs_dft.reshape(1,1)).end())-((vector<double>)abs_dft.reshape(1,1)).begin();
    cout<<"test5"<<endl;
    return ((vector<float>)ffreq_frame.reshape(1,1))[maxPosition];



}
void breath_detector::process_frame() {
    vector<float> chest_sizes={0.0}, breath_changes={0.0};
    VideoCapture cap(0);
    int x=0,y=0,w=0,h=0;
    int i=0;
    int last_time=0;
    last_time = -DETECTION_CHANGE_PERIOD;
    int frames_max = 150;
    vector<int> frames;
//    haarClassifier.load(HAAR_CLASSIFIER_PATH);
    haarClassifier.load("E:\\CLionProjects\\test\\haarcascade_frontalface_alt.xml");
    int breath = 0;
    vector<Rect> faces = {};
    Mat frameRGB,frameGray;
    while (true){
        cap.read(frameRGB);
        if (frameRGB.empty()){
            break;
        }

        cvtColor(frameRGB,frameGray,COLOR_BGR2GRAY);
        equalizeHist(frameGray,frameGray);
        i += 1;
        if(i-last_time>=DETECTION_CHANGE_PERIOD){
            haarClassifier.detectMultiScale(frameGray,faces,1.1,2,CASCADE_SCALE_IMAGE,Size(100,100));
            if(faces.size()==0){
                continue;
            }
            x = faces[0].x;
            y = faces[0].y;
            w = faces[0].width;
            h = faces[0].height;
            last_time = i;
        }
        auto face = crop_fragment(frameRGB,faces[0]);
//        frames.push_back(face_mean[1]);
        cv::Scalar face_mean;
        vector<Mat> channels1;
        split(frameRGB,channels1);

        face_mean = cv::mean(channels1[1]);
        if (frames.size()<frames_max){
            frames.push_back(face_mean[0]);
        }else{
            frames.erase(frames.begin());
            frames.push_back(face_mean[0]);
        }
        Mat gauss_out;
//        GaussianBlur(frameRGB,gauss_out,Size(7,7),0);
        GaussianBlur(frameRGB,gauss_out,Size(7,7),0);
        int aa = y + int(my_round(1.2*h));
        if(aa>480){
            aa=480;
        }
        int bb = y + int(my_round(3*h));
        if(bb>480){
            bb=480;
        }
        int cc = x - int(my_round(w*0.3));
        if(cc>640){
            cc=640;
        }
        int dd = x + int(my_round(w*1.4));
        if(dd>640){
            dd=640;
        }
//        Mat crop_img = frameRGB(Range(aa,bb),Range(cc,dd));//崩
        Mat crop_img = frameRGB(Range(aa,bb),Range(cc,dd));
        rectangle(frameRGB,Point(x,y),Point(x+w,y+h),WHITE,2);
        rectangle(frameRGB,Point(cc,aa),Point(dd,bb),GREEN,2);//胸廓区域
        rectangle(frameRGB,Point(x+floor(w/3),y+floor(h/15)),Point(x+2*floor(w/3),y+floor(h/5)),RED,2);

        //求每个通道的均值
        vector<Mat> channels2;
        split(crop_img,channels2);
        Scalar_<float16_t> r1 = mean(channels2[0]);
        Scalar_<float16_t> g1 = mean(channels2[1]);
        Scalar_<float16_t> b1 = mean(channels2[2]);


        //利用每个通道的均值建立幕布图像
        Mat tmp(crop_img.rows,crop_img.cols,CV_32FC3,Scalar(r1[0],g1[0],b1[0]));
        Mat f_crop_img;
        crop_img.convertTo(f_crop_img,CV_32FC3);
        // 胸部轮廓图像减去均值
        Mat dif_crop_img(f_crop_img.rows,crop_img.cols,CV_32FC3,Scalar(0.0,0.0,0.0));

        cv::absdiff(f_crop_img,tmp,f_crop_img);//崩 崩的原因是crop是整型，新建的tmp是float型，该函数要求类型统一
//        cv::absdiff(crop_img,tmp,dif_crop_img);

        // 对每个像素点的值累加，即构建
        vector<Mat> channels3;
        split(f_crop_img,channels3);
        Mat r2 = channels3[0];
        Mat g2 = channels3[1];
        Mat b2 = channels3[2];
        int type_r2 = r2.type();
        int type_g2 = g2.type();
        int type_b2 = b2.type();
        //python此处为np相加，超过255，add超过255的皆为255，后处理只选择<40的个数，因是否超过255 无影响
        add(r2,g2,r2);
        add(r2,b2,r2);

        //计算r2中小于40的个数即cnt
        int count_0 = countNonZero(r2);
        Mat tmp2(r2.rows,r2.cols,CV_32FC1,Scalar(39));//bug CV_32FC1改为CV_32FC3
        absdiff(r2, tmp2, r2);
        int count_1 = countNonZero(r2);
        int cnt = count_1-count_0;

        chest_sizes.push_back(float(cnt)/float(crop_img.rows*crop_img.cols));

        if(std::abs(chest_sizes[-1]-chest_sizes[-2])<0.01){
            breath_changes.push_back(breath_changes[-1]+chest_sizes[-1]+chest_sizes[-2]);
        }

        if (i%10==0){
            if (frames.size()==frames_max){
                breath = get_frequency(breath_changes,false);

            }
        }
        rectangle(frameRGB,Point(0,0),Point(400,100),WHITE,FILLED);
        string s = to_string(breath);
        putText(frameRGB,"breathing_rate"+ s,Point(10,60),FONT_ITALIC,1.1,GREEN,2);

        imshow("frame",frameRGB);
//        char ch;
//        cin>>ch;
        waitKey(1);
//        if(waitKey(1)){
//            break;
//        }
    }
}
