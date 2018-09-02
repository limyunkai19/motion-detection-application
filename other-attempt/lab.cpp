#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(){
    Mat img = imread("lena.jpg");
    Mat rgb = imread("lena.jpg");

    cvtColor(img, img, CV_BGR2Lab);

    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            int L = img.at<Vec3b>(i, j)[0];
            if(L > 200 || L < 50){
                rgb.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
            }
            else{
                img.at<Vec3b>(i, j)[0] = 150;
            }
        }
    }

    cvtColor(img, img, CV_Lab2BGR);
    imshow("lena2", img);
    waitKey(0);

    return 0;
}