#include "basic.h"
#include "opencv.h"

void getSVMParams(SVM *svm);
Ptr<SVM> svmInit(double C, double gamma);
void svmTrain(Ptr<SVM> svm, Mat &trainMat, vector<int> &trainLabels);
void svmPredict(Ptr<SVM> svm, Mat &testResponse, Mat &testMat);
void SVMevaluate(Mat &testResponse, float &count, float &accuracy, vector<int> &testLabels);
void ConvertVectortoMatrix(vector< vector<float> > &oldHOG, Mat &newMat);