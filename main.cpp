#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void mouseCallBack(int event, int x, int y, int flags, void *userdata) {
	if (event == 1) {
		printf("(%d,%d)\n", x, y);
	}
}

void createRGBHistogram(Mat src, int pixelsNumber) {
	const int n = 256, width = src.cols, heigth = src.rows;
	int reds[n] = { 0 };
	int greens[n] = { 0 };
	int blues[n] = { 0 };
	for (int x = 0; x < heigth; x++) {
		for (int y = 0; y < width; y++) {
			reds[(int)src.at<Vec3b>(x, y)[2]]++;
			greens[(int)src.at<Vec3b>(x, y)[1]]++;
			blues[(int)src.at<Vec3b>(x, y)[0]]++;
		}
	}

	int maxValueReds = 0;
	int maxValueBlues = 0;
	int maxValueGreens = 0;
	for (int i = 0; i < n; i++) {
		if (maxValueReds < reds[i]) {
			maxValueReds = reds[i];
		}
		if (maxValueBlues < blues[i]) {
			maxValueBlues = blues[i];
		}
		if (maxValueGreens < greens[i]) {
			maxValueGreens = greens[i];
		}
	}

	const int size = 500;
	Mat histR = Mat(size, size, CV_8UC3, Scalar(0));
	Mat histB = Mat(size, size, CV_8UC3, Scalar(0));
	Mat histG = Mat(size, size, CV_8UC3, Scalar(0));


	float pixelValueRedPP = size / float(maxValueReds);
	float pixelValueGreensPP = size / float(maxValueGreens);
	float pixelValueBluesPP = size / float(maxValueBlues);

	float colorValuePP = size / float(n);

	for (int i = 0; i < n; i++) {
		int x = i * colorValuePP;
		int y0 = size-1;
		int y = y0 - (reds[i] * pixelValueRedPP);
		Point p0 = Point(x, y0);
		Point p = Point(x, y);
		line(histR, p0, p, Scalar(0,0,255), (int)colorValuePP);

		y = y0 - (greens[i] * pixelValueGreensPP);
		p = Point(x, y);
		line(histG, p0, p, Scalar(0, 255, 0), (int)colorValuePP);

		y = y0 - (blues[i] * pixelValueBluesPP);
		p = Point(x, y);
		line(histB, p0, p, Scalar(255, 0, 0), (int)colorValuePP);
	}

	imshow("Histgram Red", histR);
	imshow("Histgram Green", histG);
	imshow("Histgram Blue", histB);
}

int main()
{
	Mat image = imread("sample1.jpg", IMREAD_COLOR);
	const int imagePixels = image.cols * image.rows;

	createRGBHistogram(image, imagePixels);

	namedWindow("Display Window", WINDOW_KEEPRATIO);
	imshow("Display Window", image);
	resizeWindow("Display Window", image.cols / 8, image.rows / 8);
	setMouseCallback("Display Window", mouseCallBack);

	waitKey(0);
	destroyAllWindows();
}