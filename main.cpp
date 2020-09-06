#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
#include <chrono> 
#include <ctime> 

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

double getRadius(int x0, int y0, int x, int y) {

	int ax = x - x0;
	int ay = y0 - y0;

	int bx = x - x;
	int by = y0 - y;


	double a = sqrt((ax * ax) + (ay * ay));
	double b = sqrt((bx * bx) + (by * by));

	return sqrt((a * a) + (b * b));
}

void createRGBHistogram(Mat src, double r=0) {
	const int n = 256;
	int reds[n] = { 0 };
	int greens[n] = { 0 };
	int blues[n] = { 0 };
	double newR = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (r != 0) {
				newR = getRadius(center.x, center.y, j, i);
			}
			if (newR <= r) {
				reds[(int)src.at<Vec3b>(i, j)[2]]++;
				greens[(int)src.at<Vec3b>(i, j)[1]]++;
				blues[(int)src.at<Vec3b>(i, j)[0]]++;
			}
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

Vec3b getMaskAvarage(Mat src, int size, int r, int c) {
	double sumValues[3] = { 0 };
	double countValues = 0;
	for (int row = -1 * size / 2; row <= size / 2; row++) {
		for (int col = -1 * size / 2; col <= size / 2; col++) {
			sumValues[0] += src.at<Vec3b>(r + row, c + col)[0];
			sumValues[1] += src.at<Vec3b>(r + row, c + col)[1];
			sumValues[2] += src.at<Vec3b>(r + row, c + col)[2];
			countValues++;
		}
	}
	sumValues[0] = sumValues[0] / countValues;
	sumValues[1] = sumValues[1] / countValues;
	sumValues[2] = sumValues[2] / countValues;
	return Vec3b(sumValues[0], sumValues[1], sumValues[2]);
}

Vec3b getStandarDeviation(Mat src, Vec3b average, float N, int size, int r, int c) {
	double sigma[3] = { 0 };
	double x[3] = { 0 };
	for (int row = -1 * size / 2; row <= size / 2; row++) {
		for (int col = -1 * size / 2; col <= size / 2; col++) {
			x[0] = src.at<Vec3b>(r + row, c + col)[0] - average[0];
			x[1] = src.at<Vec3b>(r + row, c + col)[1] - average[1];
			x[2] = src.at<Vec3b>(r + row, c + col)[2] - average[2];
			sigma[0] += (x[0] * x[0]);
			sigma[1] += (x[1] * x[1]);
			sigma[2] += (x[2] * x[2]);
		}
	}
	sigma[0] = sqrt(sigma[0] / N);
	sigma[1] = sqrt(sigma[1] / N);
	sigma[2] = sqrt(sigma[2] / N);
	return Vec3b(sigma[0], sigma[1], sigma[2]);
}

void mouseCallBack(int event, int x, int y, int flags, void* userdata) {
	if (event == 1) {
		mouseClicks++;
		if (mouseClicks % 2 == 0) {
			printf("Entre\n");
			Mat src = Mat(image.rows, image.cols, CV_8UC3, Scalar(0));
			double radius = getRadius(center.x, center.y, x, y);
			double nr = 0;
			const int mask_size = 9;

			for (int i = 0; i < src.rows; i++) {
				for (int j = 0; j < src.cols; j++) {
					nr = getRadius(center.x, center.y, j, i);
					if (nr > radius) {
						src.at<Vec3b>(i, j) = image.at<Vec3b>(i, j);
					}
					else {
						src.at<Vec3b>(i, j) = image.at<Vec3b>(i, j); 
						if (i - (mask_size / 2) >= 0 && i + (mask_size / 2) < src.rows && j - (mask_size / 2) >= 0 && j + (mask_size / 2) < src.cols) {
							Vec3b average = getMaskAvarage(image, mask_size, i, j);
							Vec3b sd = getStandarDeviation(image, average, (float)mask_size * mask_size, mask_size, i, j);
							src.at<Vec3b>(i, j) = sd;
						}
					}
				}
			}
			
			showImageWithAR(src, "New");
			showImageWithAR(image, "Original");
			createRGBHistogram(src, radius);
			printf("Sali\n");
		}
		else {
			center = Point(x, y);
		}
	}
}


int main()
{
	image = imread(imageName + ".jpg", IMREAD_COLOR);
	showImageWithAR(image, "Original");
	setMouseCallback(imageName + " Original", mouseCallBack);
	waitKey(0);
	destroyAllWindows();
}