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

	//�������ֱ���˳�������ֵ-1�����������·��
	cascadeName = "D:/Program Files/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";//��������·��
	nestedCascadeName = "D:/Program Files/opencv/sources/data/haarcascades/haarcascade_eye.xml";//Ƕ�׷�����·�������΢Ц��

	scale = 3;//����ѹ���ȣ�ֵԽ�����ٶ�Խ�죬����Զ�������ļ������Խ��
	if (scale < 1)
		scale = 1;
	tryflip = 0;//�Ƿ����Ƶ������

	if (!nestedCascade.load(nestedCascadeName))
		cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
	if (!cascade.load(cascadeName))
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		return -1;//û�ҵ�����������ֱ�ӽ���
	}

	if (!capture.open(0))//������ͷ��һ��Ĭ��ֵΪ0.
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

			if (waitKey(10) > 0)break;//������˳�
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
		Scalar(255,0,255)//������ע�����Ĳ�ͬ��ɫ
	};
	Mat gray, smallImg;

	cvtColor(img, gray, COLOR_BGR2GRAY);
	double fx = 1 / scale;
	resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);//����scale�ı�ԭͼ��С��scale=1.3���ǰ�ԭͼ������ֱ��Ϊԭ����1/1.3
	equalizeHist(smallImg, smallImg);//ֱ��ͼ���⻯���ú����ܹ�һ��ͼ�����Ⱥ���ǿ�Աȶ�

	t = (double)getTickCount();//��ʱ
	cascade.detectMultiScale(smallImg, faces,//faces�洢��⵽������λ�ã����Σ�
		1.25, 1, 0            
		//|CASCADE_FIND_BIGGEST_OBJECT,
		| CASCADE_DO_ROUGH_SEARCH,
		//| CASCADE_SCALE_IMAGE,
		Size(20, 20));//���ο���С��С
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
	t = (double)getTickCount() - t;//��ʱ
	printf("detection time = %g ms\tfaces detected = %d\r", t * 1000 / getTickFrequency(), faces.size());
	for (size_t i = 0; i < faces.size(); i++)//��Բ+�������
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
			center.x = cvRound((r.x + r.width*0.5)*scale);//����λ����Ϣ��Ӧ�����������Ź��ģ�����Ҫ��ԭͼ�ϻ�Բ��Ҫscale��ȥ
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
		nestedCascade.detectMultiScale(smallImgROI, nestedObjects,//������٣����ǰ�֮ǰ��⵽������������ΪROI��һ�������
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