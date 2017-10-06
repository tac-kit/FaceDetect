#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip);

string cascadeName;
string nestedCascadeName;

int main(int argc, const char** argv)
{
	VideoCapture capture;
	Mat frame, image;
	string inputName;
	bool tryflip;
	CascadeClassifier cascade, nestedCascade;
	double scale;

	//如果程序直接退出，返回值-1，请检查分类器路径
	cascadeName = "D:/Program Files/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";//主分类器路径
	nestedCascadeName = "D:/Program Files/opencv/sources/data/haarcascades/haarcascade_eye.xml";//嵌套分类器路径（检测微笑）

	scale = 3;//画面压缩比，值越大检测速度越快，但对远处人脸的检测能力越差
	if (scale < 1)
		scale = 1;
	tryflip = 0;//是否对视频镜像检测

	if (!nestedCascade.load(nestedCascadeName))
		cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
	if (!cascade.load(cascadeName))
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		return -1;//没找到分类器程序直接结束
	}

	if (!capture.open(0))//打开摄像头，一般默认值为0.
		cout << "Capture from camera 0 didn't work" << endl;

	if (capture.isOpened())
	{
		cout << "Video capturing has been started ..." << endl;

		for (;;)
		{
			capture >> frame;
			if (frame.empty())
				break;

			Mat frame1 = frame.clone();
			detectAndDraw(frame1, cascade, nestedCascade, scale, tryflip);

			if (waitKey(10) > 0)break;//任意键退出
		}
	}

	return 0;
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip)
{
	double t = 0;
	vector<Rect> faces, faces2;
	const static Scalar colors[] =
	{
		Scalar(255,0,0),
		Scalar(255,128,0),
		Scalar(255,255,0),
		Scalar(0,255,0),
		Scalar(0,128,255),
		Scalar(0,255,255),
		Scalar(0,0,255),
		Scalar(255,0,255)//用来标注人脸的不同颜色
	};
	Mat gray, smallImg;

	cvtColor(img, gray, COLOR_BGR2GRAY);
	double fx = 1 / scale;
	resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);//参数scale改变原图大小，scale=1.3就是把原图长、宽分别变为原来的1/1.3
	equalizeHist(smallImg, smallImg);//直方图均衡化，该函数能归一化图像亮度和增强对比度

	t = (double)getTickCount();//计时
	cascade.detectMultiScale(smallImg, faces,//faces存储检测到的脸的位置（矩形）
		1.25, 1, 0            
		//|CASCADE_FIND_BIGGEST_OBJECT,
		| CASCADE_DO_ROUGH_SEARCH,
		//| CASCADE_SCALE_IMAGE,
		Size(20, 20));//矩形框最小大小
	if (tryflip)
	{
		flip(smallImg, smallImg, 1);
		cascade.detectMultiScale(smallImg, faces2,
			1.1, 2, 0
			//|CASCADE_FIND_BIGGEST_OBJECT
			//|CASCADE_DO_ROUGH_SEARCH
			| CASCADE_SCALE_IMAGE,
			Size(30, 30));
		for (vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r)
		{
			faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
		}
	}
	t = (double)getTickCount() - t;//计时
	printf("detection time = %g ms\tfaces detected = %d\r", t * 1000 / getTickFrequency(), faces.size());
	for (size_t i = 0; i < faces.size(); i++)//画圆+检测器官
	{
		Rect r = faces[i];
		Mat smallImgROI;
		vector<Rect> nestedObjects;
		Point center;
		Scalar color = colors[i % 8];
		int radius;

		double aspect_ratio = (double)r.width / r.height;
		if (0.75 < aspect_ratio && aspect_ratio < 1.3)
		{
			center.x = cvRound((r.x + r.width*0.5)*scale);//人脸位置信息对应的坐标是缩放过的，所以要在原图上画圆还要scale回去
			center.y = cvRound((r.y + r.height*0.5)*scale);
			radius = cvRound((r.width + r.height)*0.25*scale);
			circle(img, center, radius, color, 2, 8, 0);
		}
		else
			rectangle(img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
				cvPoint(cvRound((r.x + r.width - 1)*scale), cvRound((r.y + r.height - 1)*scale)),
				color, 2, 8, 0);
		if (nestedCascade.empty())
			continue;
		smallImgROI = smallImg(r);
		nestedCascade.detectMultiScale(smallImgROI, nestedObjects,//检测器官，就是把之前检测到的人脸矩形作为ROI，一个个检测
			1.1, 2, 0
			//|CASCADE_FIND_BIGGEST_OBJECT
			//|CASCADE_DO_ROUGH_SEARCH
			//|CASCADE_DO_CANNY_PRUNING
			| CASCADE_SCALE_IMAGE,
			Size(5, 5));
		for (size_t j = 0; j < nestedObjects.size(); j++)
		{
			Rect nr = nestedObjects[j];
			center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
			center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
			radius = cvRound((nr.width + nr.height)*0.25*scale);
			circle(img, center, radius, color, 2, 8, 0);
		}
	}
	imshow("result", img);
	//imwrite("output.png", img);
}