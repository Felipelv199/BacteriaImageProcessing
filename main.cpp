#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
#include <chrono> 
#include <ctime> 
#include <time.h>

using namespace cv;
using namespace std;

const string imageName = "sample1";
int mouseClicks = 0, regionsPixelCount = 0, maskPixels=0;
Mat image;
Mat resized;
Mat output;
Point center;


void showImageWithAR(Mat src, String windowName) {
	const String name = windowName;
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

void createRGBHistogram(Mat src) {
	const int n = 256;
	int reds[n] = { 0 };
	int greens[n] = { 0 };
	int blues[n] = { 0 };
	double newR = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			Vec3f channels = src.at<Vec3f>(i, j);
			if (channels[0] != 0 && channels[1] != 0 && channels[2] != 0) {
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

	const int size = 256*2;
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

Mat getStandarDeviation(Mat src, int x, int y) {
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
	const double mask_size = 5;
	blur(C, C_blur, Size((int)mask_size, (int)mask_size));
	C_blur = mask.mul(C_blur);
	Mat C_dif = C - C_blur;
	Mat C_dif_pow = C_dif.mul(C_dif);
	Mat C_sigma;
	blur(C_dif_pow, C_sigma, Size((int)mask_size, (int)mask_size));
	Mat C_sd;
	sqrt(C_sigma, C_sd);
	Mat a = C_sd.clone();
	double minValue, maxValue;
	minMaxLoc(C_sd, &minValue, &maxValue);
	double range = maxValue - minValue;
	double increment = range / 256;
	a = ((256) * (C_sd - minValue)) / range;
	a = a.mul(mask);
	Mat finish_mask(mask.rows, mask.cols, a.type(), Vec3f(0, 0, 0));
	radius = radius * .92;

	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			if (mask.at<Vec3f>(i, j) != Vec3f(0, 0, 0)) {
				nr = getRadius(center.x, center.y, j, i);
				if (nr <= radius) {
					finish_mask.at<Vec3f>(i, j) = Vec3f(1.0, 1.0, 1.0);
				}
			}
		}
	}

	C_sd = finish_mask.mul(C_sd);
	vector<Mat> C_sdChannels(3);
	split(C_sd, C_sdChannels);

	for (int i = 0; i < C_sd.channels(); i++) {
		minMaxLoc(C_sdChannels[i], &minValue, &maxValue);
		range = maxValue - minValue;
		increment = range / 256;
		C_sdChannels[i] = ((256) * (C_sdChannels[i] - minValue)) / range;
	}

	merge(C_sdChannels, a);
	Mat a_b(a.rows, a.cols, CV_8UC3);
	a.convertTo(a_b, CV_8UC3);
	createRGBHistogram(a);
	Mat b;
	a.convertTo(b, CV_8UC3);
	return b;
}


double getDistances(Point2d p0, Point2d p1) {

	int x = p1.x - p0.x;
	int y = p1.y - p0.y;

	return sqrt(x * x + y * y);
}

double getColorDistance(Vec3b c0, Vec3b c1) {
	double b = c1[0]-c0[0];
	double g = c1[1]-c0[1];
	double r = c1[2]-c0[2];

	return sqrt((b*b)+(g*g)+(r*r));
}

bool growRQcolor(Mat imgori, Mat res, int x0, int y0, Vec3b val)
{
	if (x0 < 0 || x0 >= (imgori.cols) || y0 < 0 || y0 >= (imgori.rows))
	{
		return true;
	}
	Vec3b pixelImg = imgori.at<Vec3b>(y0, x0);
	std::vector<Point2d> colaPts;
	Point2d pt0(x0, y0);
	Point2d pt1;
	colaPts.clear();
	if ((res.at<Vec3b>(y0, x0)) == Vec3b(0, 0, 0))
	{
		res.at<Vec3b >(y0, x0) = val;
		regionsPixelCount++;
		for (int k = -1; k < 2; k++)
		{
			for (int l = -1; l < 2; l++)
			{
				if (!(k == 0 && l == 0)) {
					if ((x0 + l) >= 0 && (x0 + l) < (imgori.cols) && (y0 + k) >= 0 && (y0 + k) < (imgori.rows))
					{
						colaPts.push_back(Point2d((double)(x0 + l), (double)(y0 + k)));
					}
				}
			}
		}
		int counter = 0;
		while (colaPts.size() > 0)
		{
			pt1 = colaPts[0];
			colaPts.erase(colaPts.begin());
			if(getColorDistance(pixelImg, imgori.at<Vec3b>(pt1.y, pt1.x))<15)
			{
				if ((res.at<Vec3b>(pt1.y, pt1.x)) == Vec3b(0, 0, 0))
				{
					counter++;
					if (counter > 100) {
						counter = 0;
						showImageWithAR(res, "avance");
						waitKey(2);
					}
					res.at<Vec3b>(pt1.y, pt1.x) = val;
					regionsPixelCount++;
					for (int k = -1; k < 2; k++)
					{
						for (int l = -1; l < 2; l++)
						{
							if (!(k == 0 && l == 0))
								if ((pt1.x + l) >= 0 && (pt1.x + l) < (imgori.cols) && (pt1.y + k) >= 0 && (pt1.y + k) < (imgori.rows))
								{
									if ((res.at<Vec3b>(pt1.y + k, pt1.x + l)) == Vec3b(0, 0, 0))
										colaPts.push_back(Point2d(pt1.x + l, pt1.y + k));
								}
						}
					}
				}
			}
		}
		showImageWithAR(res, "avance");
	}
	return false;
}

void mouseCallBack_Regions(int event, int x, int y, int flags, void* userdata) {
	if (event == 1) {
		Point point = Point(x, y);
		Vec3b pixel = Vec3b(255,255,255);
		growRQcolor(resized, output, x, y, pixel);
		printf("Circle pixel number area: %d\n", maskPixels);
		printf("Number of white pixels: %d\n", regionsPixelCount);
		printf("White pixels circle area percentage: %f%%\n", (regionsPixelCount/(double)(maskPixels))*100);
		showImageWithAR(resized, "mask");
	}
}

void mouseCallBack_Mask(int event, int x, int y, int flags, void* userdata) {
	if (event == 1) {
		mouseClicks++;
		if (mouseClicks % 2 == 0) {
			printf("Entre\n");
			regionsPixelCount = 0;
			Mat mask_inside = Mat(resized.rows, resized.cols, resized.type(), Scalar(0));
			double radius = getRadius(center.x, center.y, x, y);
			double nr = 0;
			maskPixels = 0;
			for (int i = 0; i < resized.rows; i++) {
				for (int j = 0; j < resized.cols; j++) {
					nr = getRadius(center.x, center.y, j, i);
					if (nr <= radius) {
						mask_inside.at<Vec3b>(i, j) = Vec3b(1,1,1);
						maskPixels++;
					}
				}
			}
			resized = mask_inside.mul(resized);
			showImageWithAR(resized, "mask");
			setMouseCallback("mask", mouseCallBack_Regions);
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
	int scalar = 2;
	resized = Mat(image.rows/scalar, image.cols/scalar, image.type());
	resize(image, resized, resized.size(),0.5,0.5,INTER_LINEAR);
	output  = Mat(resized.rows, resized.cols, image.type(), Scalar(0, 0, 0));
	showImageWithAR(resized, "Normalized");
	setMouseCallback("Normalized", mouseCallBack_Mask);
	waitKey(0);
	destroyAllWindows();
}