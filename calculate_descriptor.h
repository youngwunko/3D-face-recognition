#include "basic.h"
#include "opencv.h"

vector<float> calc_descriptor(Mat rgb_image, Mat depth_image, bool flag); // �� Ư¡ �����ϴ� �Լ�
float calcEntropy(Mat image, int i, int j, int flag); // �̹����� entropy ����ϴ� �Լ�