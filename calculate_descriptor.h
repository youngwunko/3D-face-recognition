#include "basic.h"
#include "opencv.h"

vector<float> calc_descriptor(Mat rgb_image, Mat depth_image, bool flag); // 얼굴 특징 추출하는 함수
float calcEntropy(Mat image, int i, int j, int flag); // 이미지의 entropy 계산하는 함수