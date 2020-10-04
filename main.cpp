#pragma warning(disable: 4996)
#include <Windows.h>
#include <NuiApi.h>
#include <NuiImageCamera.h>
#include <NuiSensor.h>
#include "opencv.h"
#include "calculate_descriptor.h"
#include "svm.h"

// Kinect ����
INuiSensor* sensor;
HANDLE rgbStream;
HANDLE depthStream;

bool initKinect();
Mat getKinectRGBData();
Mat getKinectDepthData();

int main(void) {
	int choice = 1;
	Mat rgb_image, depth_image;
	int descriptor_size = 10764;
	double C = 312.5, gamma = 0.00015;
	Ptr<SVM> model = svmInit(C, gamma);
		
	//////////////////    SVM ���� �Ʒ� ��� �ҷ�����  /////////////////
	vector< vector<float> > trainHOG;
	vector<int> trainLabels;
	ifstream descriptor_stream("descriptor_db.txt");
	while (1) {
		int name_num;
		float num;
		vector<float> final_descriptor;
		descriptor_stream >> name_num;
		if (descriptor_stream.eof())
			break;

		for (int h = 0; h < 10764; h++)	{
			descriptor_stream >> num;
			final_descriptor.push_back(num);
		}
		trainHOG.push_back(final_descriptor);
		trainLabels.push_back(name_num);
	}
	descriptor_stream.close();

	ifstream name_stream("name_db.txt");
	vector< pair<int, string> > name_db;
	while (1) {
		pair<int, string> temp;
		name_stream >> temp.first >> temp.second;
		if (name_stream.eof())
			break;
		name_db.push_back(temp);
	}
	name_stream.close();
	////////////////////////////////////////////////////////

	if (!initKinect()) {
		printf("Kinect ���� ����\n");
	}
	else {
		while (choice) {			
			printf("1 : �� ��� / 2 : �� �ν� / -1 : ����\n");
			scanf("%d", &choice);

			if (choice == -1) {
				break;
			}
			else if (choice == 3) {
				system("cls");
			}
			else {
				if (choice == 1) { // registration (training)				
					int name_num = 0;
					string name_temp;
					cout << "����� ����� �̸��� �Է��ϼ���\n";
					cin >> name_temp;

					rgb_image = getKinectRGBData();
					depth_image = getKinectDepthData();

					vector<float> final_descriptor = calc_descriptor(rgb_image, depth_image, 0);

					if (final_descriptor.size() == 0) {
						cout << "�� Ž�� ���� - �ٽ� �õ��ϼ���" << endl << endl;
					}
					else if (final_descriptor.size() == 1) {
						cout << "���� �󱼷� �ٽ� �õ��ϼ���" << endl << endl;
					}
					else if (final_descriptor.size() > 1) {					
						for (int i = 0; i < (int)name_db.size(); i++) {
							if (name_db.at(i).second == name_temp)
								name_num = name_db.at(i).first;
						}
						if (name_num == 0) { // ���ο� ����� ��						
							name_num = name_db.size() + 1;

							// ���ο� �̸� �߰�
							pair<int, string> temp;
							temp.first = name_num;
							temp.second = name_temp;
							name_db.push_back(temp);

							ofstream name_stream("name_db.txt", ios::app);
							name_stream << temp.first << " " << temp.second << "\n";
							name_stream.close();
						}
						
						// ���ο� descriptor �߰�
						ofstream descriptor_stream("descriptor_db.txt", ios::app);
						descriptor_stream << name_num << " ";
						for (int h = 0; h < (int)final_descriptor.size(); h++)
							descriptor_stream << final_descriptor.at(h) << " ";
						descriptor_stream << "\n";
						descriptor_stream.close();

						trainHOG.push_back(final_descriptor);
						trainLabels.push_back(name_num);

						Mat trainMat(trainHOG.size(), descriptor_size, CV_32FC1);
						ConvertVectortoMatrix(trainHOG, trainMat);
						svmTrain(model, trainMat, trainLabels);
						printf("��� �Ϸ�\n\n");
					}
				}
				else if (choice == 2) { // recognition (testing)				
					vector< vector<float> > testHOG;
					Mat testResponse;

					rgb_image = getKinectRGBData();
					depth_image = getKinectDepthData();

					vector<float> final_descriptor = calc_descriptor(rgb_image, depth_image, 1);

					if (final_descriptor.size() == 0) {
						cout << "�� Ž�� ���� - �ٽ� �õ��ϼ���" << endl << endl;
					}						
					else if (final_descriptor.size() == 1) {
						cout << "���� �󱼷� �ٽ� �õ��ϼ���" << endl << endl;
					}
					else if (final_descriptor.size() > 1) {
						testHOG.push_back(final_descriptor);
						Mat testMat(testHOG.size(), descriptor_size, CV_32FC1);
						ConvertVectortoMatrix(testHOG, testMat);
						svmPredict(model, testResponse, testMat);

						for (int i = 0; i < (int)name_db.size(); i++) {
							if (name_db.at(i).first == testResponse.at<float>(0, 0))
								cout << "�ν� ��� : " << name_db.at(i).second << endl << endl;
						}
					}
				}
			}
		}
	}
	return 0;
}

bool initKinect() {
	int numSensors;
	if (NuiGetSensorCount(&numSensors) < 0 || numSensors < 1) return false;
	if (NuiCreateSensorByIndex(0, &sensor) < 0) return false;

	sensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_COLOR | NUI_INITIALIZE_FLAG_USES_DEPTH);
	sensor->NuiImageStreamOpen(NUI_IMAGE_TYPE_COLOR, NUI_IMAGE_RESOLUTION_640x480, 0, 2, NULL, &rgbStream);
	sensor->NuiImageStreamOpen(NUI_IMAGE_TYPE_DEPTH, NUI_IMAGE_RESOLUTION_640x480, 0, 2, NULL, &depthStream);
	return sensor;
}

Mat getKinectRGBData() {
	Mat rgb_image = Mat::zeros(480, 640, CV_8UC4);

	NUI_IMAGE_FRAME imageFrame;
	NUI_LOCKED_RECT LockedRect;
	if (sensor->NuiImageStreamGetNextFrame(rgbStream, 0, &imageFrame) < 0)
		printf("Error - read RGB data\n");

	INuiFrameTexture* texture = imageFrame.pFrameTexture;
	texture->LockRect(0, &LockedRect, NULL, 0);
	if (LockedRect.Pitch != 0) {
		const BYTE* curr = (const BYTE*)LockedRect.pBits;

		for (int i = 0; i < rgb_image.rows; i++) { // ���� �ǹ�		
			for (int j = 0; j < rgb_image.cols; j++) { // ���� �ǹ�			
				rgb_image.at<Vec4b>(i, j)[0] = *curr++;
				rgb_image.at<Vec4b>(i, j)[1] = *curr++;
				rgb_image.at<Vec4b>(i, j)[2] = *curr++;
				rgb_image.at<Vec4b>(i, j)[3] = *curr++;
			}
		}
	}
	texture->UnlockRect(0);
	sensor->NuiImageStreamReleaseFrame(rgbStream, &imageFrame);
	texture->Release();
	return rgb_image;
}

Mat getKinectDepthData() {
	Mat depth_image = Mat::zeros(480, 640, CV_16U);

	NUI_IMAGE_FRAME imageFrame;
	NUI_LOCKED_RECT LockedRect;
	if (sensor->NuiImageStreamGetNextFrame(depthStream, 0, &imageFrame) < 0)
		printf("Error - read depth data\n");

	INuiFrameTexture* texture = imageFrame.pFrameTexture;
	texture->LockRect(0, &LockedRect, NULL, 0);
	if (LockedRect.Pitch != 0) {
		const USHORT* curr = (const USHORT*)LockedRect.pBits;

		for (int i = 0; i < depth_image.rows; i++) { // ���� �ǹ�		
			for (int j = 0; j < depth_image.cols; j++) { // ���� �ǹ�			
				USHORT depth = NuiDepthPixelToDepth(*curr++);
				depth_image.at<ushort>(i, j) = depth;
			}
		}
	}
	texture->UnlockRect(0);
	sensor->NuiImageStreamReleaseFrame(depthStream, &imageFrame);
	texture->Release();
	return depth_image;
}