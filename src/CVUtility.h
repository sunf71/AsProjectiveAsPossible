#pragma once
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <Eigen/Dense>

void combineMat(cv::Mat &out, const cv::Mat& left, const cv::Mat& right);
void displayMat(const cv::Mat& display);
void getColorSubpixelRGB(const cv::Mat &image, float x, float y, int width, int height, uint8_t& r, uint8_t& g, uint8_t& b);
void detectSiftMatchWithOpenCV(const char* img1_path, const char* img2_path, Eigen::MatrixXf &match);
void detectKLTMatch(const char* img1_path, const char* img2_path, Eigen::MatrixXf &match);
void KLTFeaturesMatching(const cv::Mat& simg, const cv::Mat& timg, std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2, int cornerCount = 100, float featureC = 0.05, float minDist = 10); 
