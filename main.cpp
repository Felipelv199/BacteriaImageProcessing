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

int main()
{
	Mat image = imread("sample1.jpg");
	namedWindow("Display Window", WINDOW_KEEPRATIO);
	imshow("Display Window", image);
	resizeWindow("Display Window", image.cols / 8, image.rows / 8);
	setMouseCallback("Display Window", mouseCallBack);
	waitKey(0);
}