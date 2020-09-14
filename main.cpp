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
const string imageName = "sample2";


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

void createRGBHistogram(Mat src, double r=0, int scale=1) {
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
				Vec3f channels = src.at<Vec3b>(i, j);
				reds[(int)channels[2]]++;
				greens[(int)channels[1]]++;
				blues[(int)channels[0]]++;
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

	const int size = 256*scale;
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

Vec3f getStandarDeviation(Mat src, int size, int r, int c) {
	float sigma[3] = {0};

	for (int row = -1 * size / 2; row <= size / 2; row++) {
		for (int col = -1 * size / 2; col <= size / 2; col++) {
			Vec3f x = src.at<Vec3f>(r + row, c + col);
			sigma[0] += x[0];
			sigma[1] += x[1];
			sigma[2] += x[2];
		}
	}

	sigma[0] = sqrt(sigma[0]);
	sigma[1] = sqrt(sigma[1]);
	sigma[2] = sqrt(sigma[2]);
	return Vec3f(sigma[0], sigma[1], sigma[2]);
}

void mouseCallBack(int event, int x, int y, int flags, void* userdata) {
	if (event == 1) {
		mouseClicks++;
		if (mouseClicks % 2 == 0) {
			printf("Entre\n");
			Mat image_float;
			image.convertTo(image_float, CV_32FC3);
			Mat mask = Mat(image.rows, image.cols, CV_32FC3, Scalar(0));
			double radius = getRadius(center.x, center.y, x, y);
			double nr = 0;

			for (int i = 0; i < mask.rows; i++) {
				for (int j = 0; j < mask.cols; j++) {
					nr = getRadius(center.x, center.y, j, i);
					if (nr <= radius) {
						mask.at<Vec3f>(i, j) = Vec3f(1.0, 1.0, 1.0);
					}
				}
			}

			Mat C = image_float.mul(mask);
			Mat C_blur = Mat(image.rows, image.cols, CV_32FC3, Scalar(0));
			const double mask_size = 9;
			blur(C, C_blur, Size((int)mask_size, (int)mask_size));
			Mat C_dif = C - C_blur;
			Mat C_dif_pow = C_dif.mul(C_dif/ (mask_size * mask_size));

			Mat C_sd = Mat(image.rows, image.cols, CV_32FC3, Scalar(0));
			for (int i = 0; i < C_sd.rows; i++) {
				for (int j = 0; j < C_sd.cols; j++) {
					nr = getRadius(center.x, center.y, j, i);
					if (nr <= radius) {
						if (i - (mask_size / 2) >= 0 && i + (mask_size / 2) < C_dif_pow.rows && j - (mask_size / 2) >= 0 && j + (mask_size / 2) < C_dif_pow.cols) {
							 Vec3f sd = getStandarDeviation(C_dif_pow, (int)mask_size, i, j);
							 C_sd.at<Vec3f>(i, j) = sd;
						}
					}
				}
			}

			Mat outputImage;
			C_sd.convertTo(outputImage, CV_8UC3);
			showImageWithAR(outputImage, "New");
			showImageWithAR(image, "Original");
			imwrite("image2.png", outputImage);
			createRGBHistogram(outputImage, radius, 2);
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