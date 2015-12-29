#include "CVUtility.h"
#include <vector>


using namespace cv;
using namespace Eigen;
using namespace std;

void getColorSubpixelRGB(const Mat &image, float x, float y, int width, int height, uint8_t& r, uint8_t& g, uint8_t& b) {
    int x_int = (int)x;
    int y_int = (int)y;
    
    int x0 = x_int<0?0:(x_int>=width?width-1:x_int);
    int x1 = x_int+1<0?0:(x_int+1>=width?width-1:x_int+1);
    int y0 = y_int<0?0:(y_int>=height?height-1:y_int);
    int y1 = y_int+1<0?0:(y_int+1>=height?height-1:y_int+1);
    
    float dx = x - (float)x_int;
    float dy = y - (float)y_int;
    
    r = (1.f-dy)*(image.at<Vec3b>(y0, x0)[0]*(1.f-dx)+image.at<Vec3b>(y0, x1)[0]*dx)+dy*(image.at<Vec3b>(y1, x0)[0]*(1.f-dx)+image.at<Vec3b>(y1, x1)[0]*dx);
    g = (1.f-dy)*(image.at<Vec3b>(y0, x0)[1]*(1.f-dx)+image.at<Vec3b>(y0, x1)[1]*dx)+dy*(image.at<Vec3b>(y1, x0)[1]*(1.f-dx)+image.at<Vec3b>(y1, x1)[1]*dx);
    b = (1.f-dy)*(image.at<Vec3b>(y0, x0)[2]*(1.f-dx)+image.at<Vec3b>(y0, x1)[2]*dx)+dy*(image.at<Vec3b>(y1, x0)[2]*(1.f-dx)+image.at<Vec3b>(y1, x1)[2]*dx);
}

void combineMat(Mat &out, const Mat& left, const Mat& right) {
  int height = left.size[0];
  int width = left.size[1];
  out = Mat(height, 2*width, CV_8UC3);
  for (int i = 0; i < height; i++) 
    for (int j = 0; j < width*2; j++) 
      if (j < width)
        out.at<Vec3b>(i, j) = left.at<Vec3b>(i, j);
      else
        out.at<Vec3b>(i, j) = right.at<Vec3b>(i, j-width);
}

void displayMat(const Mat& display) {
  int height = display.size[0];
  int width = display.size[1];
  int longEdge = max(height, width);
  float resize_ratio = 1000.f/longEdge;
  Size size((int)width*resize_ratio, (int)height*resize_ratio);
  Mat resized;
  resize(display, resized, size);
  imshow("img", resized);
  waitKey(0);
}

void detectSiftMatchWithOpenCV(const char* img1_path, const char* img2_path, MatrixXf &match) {
  Mat img1 = imread(img1_path);   
  Mat img2 = imread(img2_path);   

  SiftFeatureDetector detector;
  SiftDescriptorExtractor extractor;
  vector<KeyPoint> key1;
  vector<KeyPoint> key2;
  Mat desc1, desc2;
  detector.detect(img1, key1);
  detector.detect(img2, key2);
  extractor.compute(img1, key1, desc1);
  extractor.compute(img2, key2, desc2);

  FlannBasedMatcher matcher;
  vector<DMatch> matches;
  matcher.match(desc1, desc2, matches);

  match.resize(matches.size(), 6);
  cout << "match count: " << matches.size() << endl;
  for (int i = 0; i < matches.size(); i++) {
    match(i, 0) = key1[matches[i].queryIdx].pt.x;
    match(i, 1) = key1[matches[i].queryIdx].pt.y;
    match(i, 2) = 1;
    match(i, 3) = key2[matches[i].trainIdx].pt.x;
    match(i, 4) = key2[matches[i].trainIdx].pt.y;
    match(i, 5) = 1;
  }
  
}

void KLTFeaturesMatching(const cv::Mat& simg, const cv::Mat& timg, std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2, int cornerCount, float quality, float minDist)
{
	vf1.clear();
	vf2.clear();
	std::vector<uchar>status;
	std::vector<float> err;
	cv::Mat sGray, tGray;
	if (simg.channels() == 3)
		cv::cvtColor(simg, sGray, CV_BGR2GRAY);
	else
		sGray = simg;
	if (timg.channels() == 3)
		cv::cvtColor(timg, tGray, CV_BGR2GRAY);
	else
		tGray = timg;
	cv::goodFeaturesToTrack(sGray, vf1, cornerCount, quality, minDist);
	cv::calcOpticalFlowPyrLK(sGray, tGray, vf1, vf2, status, err);
	int k = 0;
	for (int i = 0; i<vf1.size(); i++)
	{
		if (status[i] == 1)
		{
			vf1[k] = vf1[i];
			vf2[k] = vf2[i];
			k++;
		}
	}

	vf1.resize(k);
	vf2.resize(k);
	//FeaturePointsRefineHistogram(vf1,vf2);
}


void detectKLTMatch(const char* img1_path, const char* img2_path, Eigen::MatrixXf &match)
{
	Mat img1 = imread(img1_path);
	Mat img2 = imread(img2_path);
	if (img1.channels() == 3)
	{
		cvtColor(img1, img1, CV_BGR2GRAY);
		cvtColor(img2, img2, CV_BGR2GRAY);
	}
	vector<Point2f> f1, f2;
	KLTFeaturesMatching(img1, img2, f1, f2, 500);

	match.resize(f1.size(), 6);
	for (int i = 0; i < f1.size(); i++) {
		match(i, 0) = f1[i].x;
		match(i, 1) = f1[i].y;
		match(i, 2) = 1;
		match(i, 3) = f2[i].x;
		match(i, 4) = f2[i].y;
		match(i, 5) = 1;
	}
}