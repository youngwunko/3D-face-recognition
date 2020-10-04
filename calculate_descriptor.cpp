#include "calculate_descriptor.h"

vector<float> calc_descriptor(Mat rgb_image, Mat depth_image, bool flag) {
	///////////////////////    Face detection    //////////////////////////////////
	CascadeClassifier face_detection("haarcascade_frontalface_default.xml");
	vector<Rect> detected; // 검출된 얼굴 이미지
	face_detection.detectMultiScale(rgb_image, detected);
	if (detected.size() == 0) {
		vector<float> err = {};
		return err;
	}
	Mat face;
	int face_num = 0;
	if (detected.size() > 1)
		face_num = 1;

	resize(rgb_image(detected.at(face_num)), face, Size(96, 96));

	// Depth와 RGB 이미지가 캡처될 때 위치가 달라 맞춰주는 코드 (캘리브레이션할 수 있게 변경하기)
	Mat face_depth;
	Rect a;
	a.x = detected.at(face_num).x - 15;
	a.y = detected.at(face_num).y - 30;
	a.width = detected.at(face_num).width;
	a.height = detected.at(face_num).height;

	resize(depth_image(a), face_depth, Size(96, 96));
	////////////////////////////////////////////////////////////////////////////////////////////


	Rect half((face.cols / 4), (face.rows / 4), (face.cols / 2), (face.rows / 2));
	Rect quarter_3((face.cols / 8), (face.rows / 8), (face.cols * 3 / 4), (face.rows * 3 / 4));

	////////////////////    RGB Entropy map     /////////////////////////////////
	Mat image1 = face;
	cvtColor(face, image1, COLOR_BGR2GRAY);
	Mat rgb_entropy_map(image1.rows, image1.cols, CV_32F);

	for (int i = 0; i < image1.rows; i++) { // 열을 의미	
		for (int j = 0; j < image1.cols; j++) { // 행을 의미		
			rgb_entropy_map.at<float>(i, j) = calcEntropy(image1, j, i, 0);
		}
	}
	normalize(rgb_entropy_map, rgb_entropy_map, 0, 255, NORM_MINMAX, CV_8U);

	Mat rgb_entropy_map1 = rgb_entropy_map(half);
	Mat rgb_entropy_map2 = rgb_entropy_map(quarter_3);
	///////////////////////////////////////////////////////////////////////////////


	////////////////////    Depth Entropy map     /////////////////////////////////
	Mat image2 = face_depth;
	Mat depth_entropy_map(image2.rows, image2.cols, CV_32F);

	float sum = 0; // 사진과 실제 얼굴을 구분하기 위한 변수
	for (int i = 0; i < image2.rows; i++) { // 열을 의미	
		for (int j = 0; j < image2.cols; j++) { // 행을 의미		
			depth_entropy_map.at<float>(i, j) = calcEntropy(image2, j, i, 1);
			sum += calcEntropy(image2, j, i, 1);
		}
	}
	//printf("Sum : %f\n", sum);	
	normalize(depth_entropy_map, depth_entropy_map, 0, 255, NORM_MINMAX, CV_8U);

	Mat depth_entropy_map1 = depth_entropy_map(half);
	Mat depth_entropy_map2 = depth_entropy_map(quarter_3);
	///////////////////////////////////////////////////////////////////////////////


	/////////////////////     Saliency map     ////////////////////////////////
	Ptr<StaticSaliencySpectralResidual> saliencyAlgorithm = StaticSaliencySpectralResidual::create();
	Mat saliency_map;
	Mat saliency_temp = face;
	cvtColor(saliency_temp, saliency_temp, COLOR_BGR2GRAY);
	saliencyAlgorithm->computeSaliency(saliency_temp, saliency_map);
	normalize(saliency_map, saliency_map, 0, 255, NORM_MINMAX, CV_8U);
	/////////////////////////////////////////////////////////////////////////


	//////////////////////////////////////////       HOG        ////////////////////////////////////////////	
	HOGDescriptor hog;
	vector<float> rgb_hog1, rgb_hog2, depth_hog1, depth_hog2, saliency_hog;
	vector<float> final_descriptor;

	hog.winSize = Size(48, 48);
	hog.compute(rgb_entropy_map1, rgb_hog1);
	hog.compute(depth_entropy_map1, depth_hog1);

	hog.winSize = Size(72, 72);
	hog.compute(rgb_entropy_map2, rgb_hog2);
	hog.compute(depth_entropy_map2, depth_hog2);

	hog.winSize = Size(96, 96);
	hog.compute(saliency_map, saliency_hog);

	final_descriptor = rgb_hog1;
	final_descriptor.insert(final_descriptor.end(), rgb_hog2.begin(), rgb_hog2.end());
	final_descriptor.insert(final_descriptor.end(), depth_hog1.begin(), depth_hog1.end());
	final_descriptor.insert(final_descriptor.end(), depth_hog2.begin(), depth_hog2.end());
	final_descriptor.insert(final_descriptor.end(), saliency_hog.begin(), saliency_hog.end());

	//////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////   실행 결과 출력      ////////////////////////////////////////
	Mat rgb_depth;
	resize(rgb_image, rgb_image, Size(480, 360));
	resize(depth_image, depth_image, Size(480, 360));
	cvtColor(rgb_image, rgb_image, COLOR_RGBA2RGB);
	normalize(depth_image, depth_image, 0, 255, NORM_MINMAX, CV_8U);
	cvtColor(depth_image, depth_image, COLOR_GRAY2RGB);
	hconcat(rgb_image, depth_image, rgb_depth);
	imshow("RGB & Depth image", rgb_depth);

	Mat entropy_rgb_depth;
	resize(rgb_entropy_map, rgb_entropy_map, Size(200, 200));
	resize(depth_entropy_map, depth_entropy_map, Size(200, 200));
	hconcat(rgb_entropy_map, depth_entropy_map, entropy_rgb_depth);
	imshow("Entropy RGB & Depth image", entropy_rgb_depth);

	resize(saliency_map, saliency_map, Size(200, 200));
	imshow("Saliency map", saliency_map);

	moveWindow("RGB & Depth image", 0, 0);
	moveWindow("Entropy RGB & Depth image", 0, 390);
	moveWindow("Saliency map", 400, 390);
	waitKey(0);
	//////////////////////////////////////////////////////////////////////////////////////////////////

	if (sum < 70000) {  // 사진일 경우 에러값 반환
		vector<float> err = { 1.0 };
		return err;
	}
	else
		return final_descriptor;
}

float calcEntropy(Mat image, int i, int j, int flag) {
	float entropy = 0;
	int pixel_value = 0;
	float occur = 0;
	int histSize = 4096;
	float range[] = { 0, 4096 };
	const float* histRange = { range };
	bool uniform = true, accumulate = false;

	// 원래 image size 벗어나는 경우 예외 처리
	int poisx = i - 2, poisy = j - 2; // rect 만들 때 시작 좌표
	int sizex = 5, sizey = 5;

	if (i - 2 < 0) {
		poisx = 0;
		sizex -= (2 - i);
	}
	if (j - 2 < 0) {
		poisy = 0;
		sizey -= (2 - j);
	}
	if (i + 2 >= image.cols) {
		sizex -= (3 - (image.cols - i));
	}
	if (j + 2 >= image.rows) {
		sizey -= (3 - (image.rows - j));
	}
	Rect r(poisx, poisy, sizex, sizey);
	Mat neighbor = image(r);

	Mat occur_arr; // pixel의 histogram
	calcHist(&neighbor, 1, 0, Mat(), occur_arr, 1, &histSize, &histRange, uniform, accumulate);

	for (int m = 0; m < neighbor.rows; m++) {
		for (int n = 0; n < neighbor.cols; n++) {
			if (flag == 0)
				pixel_value = neighbor.at<uchar>(m, n);
			else
				pixel_value = neighbor.at<ushort>(m, n);
			occur = occur_arr.at<float>(pixel_value, 0);

			entropy -= (occur / neighbor.total()) * (log2(occur / neighbor.total()));
		}
	}
	return entropy;
}