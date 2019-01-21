#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);
void reduceVector(vector<cv::Point3f> &v, vector<uchar> status);

inline void computeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C);


class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img, const cv::Mat &_depth, double _cur_time);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();
	// use PnP to reject depth outliers
	void rejectWithPnP();
    // use Sim3 to reject depth outliers
    void rejectWithSim3();
    bool computeSim3(cv::Mat &P1, cv::Mat &P2, cv::Mat &mR12i, cv::Mat &mt12i);
    int checkInliers(cv::Mat mR12i, cv::Mat mt12i, vector<uchar> &mvbInliersi);
    void project(vector<cv::Point2f> &vP2D, cv::Mat mR12i, cv::Mat mt12i);

    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img, forw_img;
    cv::Mat prev_depth, cur_depth, forw_depth;

    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts;
    vector<cv::Point2f> pts_velocity;
    vector<cv::Point3f> mvX3Dc1;
    vector<cv::Point3f> mvX3Dc2;
    vector<int> ids;
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    camodocal::CameraPtr m_camera;
    double cur_time;
    double prev_time;

    static int n_id;
};
