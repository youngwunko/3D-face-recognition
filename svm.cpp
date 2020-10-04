#include "svm.h"

void getSVMParams(SVM *svm) {
	cout << "Kernel type     : " << svm->getKernelType() << endl;
	cout << "Type            : " << svm->getType() << endl;
	cout << "C               : " << svm->getC() << endl;
	cout << "Degree          : " << svm->getDegree() << endl;
	cout << "Nu              : " << svm->getNu() << endl;
	cout << "Gamma           : " << svm->getGamma() << endl;
}

Ptr<SVM> svmInit(double C, double gamma) {
	Ptr<SVM> svm = SVM::create();
	svm->setGamma(gamma);
	svm->setC(C);
	svm->setKernel(SVM::RBF);
	svm->setType(SVM::C_SVC);
	return svm;
}

void svmTrain(Ptr<SVM> svm, Mat &trainMat, vector<int> &trainLabels) {
	Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);
	svm->train(td);
}

void svmPredict(Ptr<SVM> svm, Mat &testResponse, Mat &testMat) {
	svm->predict(testMat, testResponse);
}

void SVMevaluate(Mat &testResponse, float &count, float &accuracy, vector<int> &testLabels) {
	cout << "<Test 결과>" << endl;
	cout << "대상      예측결과" << endl;
	for (int i = 0; i < testResponse.rows; i++) {
		cout << testLabels[i] << "		" << testResponse.at<float>(i, 0) << endl;
		if (testResponse.at<float>(i, 0) == testLabels[i])
			count = count + 1;
	}
	accuracy = (count / testResponse.rows) * 100;
}

void ConvertVectortoMatrix(vector< vector<float> > &oldHOG, Mat &newMat) {
	int descriptor_size = oldHOG[0].size();

	for (int i = 0; i < (int)oldHOG.size(); i++) {
		for (int j = 0; j < descriptor_size; j++) {
			newMat.at<float>(i, j) = oldHOG[i][j];
		}
	}
}