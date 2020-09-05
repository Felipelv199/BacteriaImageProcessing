#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

Mat image;
int mouseClicks = 0;
Point center;
const string imageName = "sample1";


void showImageWithAR(Mat src, String windowName) {
	const String name = imageName + " " + windowName;
	namedWindow(name, WINDOW_KEEPRATIO);
	imshow(name, src);
	resizeWindow(name, src.cols / 8, src.rows / 8);
}

void createRGBHistogram(Mat src) {
	const int pixelsNumber = src.cols * src.rows;
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

double getRadius(int x0, int y0, int x, int y) {

	int ax = x - x0;
	int ay = y0 - y0;

	int bx = x - x;
	int by = y0 - y;


	double a = sqrt((ax * ax) + (ay * ay));
	double b = sqrt((bx * bx) + (by * by));

	return sqrt((a * a) + (b * b));
}

double getMaskAvarage(Mat src, int size, int r, int c) {
	int sumValues = 0;
	double countValues = 0;
	for (int row = -1 * size / 2; row <= size / 2; row++) {
		for (int col = -1 * size / 2; col <= size / 2; col++) {
			sumValues += (int)src.at<uchar>(r + row, c + col);
			countValues++;
		}
	}
	return sumValues / countValues;
}

double getStandarDeviation(Mat src, double average, float N, int size, int r, int c) {
	double sigma = 0;
	double x = 0;
	for (int row = -1 * size / 2; row <= size / 2; row++) {
		for (int col = -1 * size / 2; col <= size / 2; col++) {
			x = (int)src.at<uchar>(r + row, c + col) - average;
			sigma += (x * x);
		}
	}
	return sqrt(sigma / N);
}

void imageStandardDevation(Mat src) {
	const int mask_size = 9;
	for (int i = mask_size / 2; i < src.rows - mask_size / 2; i++) {
		for (int j = mask_size / 2; j < src.cols - mask_size / 2; j++) {
			if ((int)src.at<uchar>(i, j) != 0) {
				if (i - (mask_size / 2) >= 0 && i + (mask_size / 2) < src.rows && j - (mask_size / 2) >= 0 && j + (mask_size / 2) < src.cols) {
					double average = getMaskAvarage(src, mask_size, i, j);
					double sd = getStandarDeviation(src, average, (float)mask_size*mask_size, mask_size, i, j);
					src.at<uchar>(i, j) = (int)sd;
				}
			}
		}

	}
}

void mouseCallBack(int event, int x, int y, int flags, void* userdata) {
	if (event == 1) {
		mouseClicks++;
		if (mouseClicks % 2 == 0) {
			printf("Entre\n");
			Mat src = Mat(image.rows, image.cols, CV_8UC1, Scalar(0));
			int heigth = src.rows, width = src.cols;
			double radius = getRadius(center.x, center.y, x, y);
			int nx = 0, ny = 0;
			double nr = 0;
			for (int i = 0; i < src.rows; i++) {
				for (int j = 0; j < src.cols; j++) {
					nr = getRadius(center.x, center.y, j, i);
					if (nr > radius) {
						src.at<uchar>(i, j) = 0;
					}
					else {
						src.at<uchar>(i, j) = image.at<uchar>(i, j);
					}
					
				}
			}

			imageStandardDevation(src);
			showImageWithAR(src, "New");
			showImageWithAR(image, "Original");
			//createRGBHistogram(image);
			printf("Sali\n");
		}
		else {
			center = Point(x, y);
		}
	}
}


int main()
{
	image = imread(imageName + ".jpg", IMREAD_GRAYSCALE);
	showImageWithAR(image, "Original");
	setMouseCallback(imageName + " Original", mouseCallBack);

	waitKey(0);
	destroyAllWindows();
}