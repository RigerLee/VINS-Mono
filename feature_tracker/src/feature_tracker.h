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



class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img, const cv::Mat &_color_depth, const cv::Mat &_depth, double _cur_time);

    void setMask();
	void setMaskDepth();

    void addPoints();

    bool updateID(unsigned int i);
	bool updateIDDepth(unsigned int i);

    void readIntrinsicParameter(const string &calib_file, const string &depth_calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();
	void rejectWithFDepth();

    void undistortedPoints();
    void undistortedPointsDepth();
	// use PnP to reject depth outliers
	void rejectWithPnP();

    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img, forw_img;
    cv::Mat prev_depth, cur_depth, forw_depth;
    //newly added
    cv::Mat prev_color_depth, cur_color_depth, forw_color_depth;

    vector<cv::Point2f> n_pts;
	vector<cv::Point2f> n_pts_depth;
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    //newly added
	vector<cv::Point2f> prev_pts_depth, cur_pts_depth, forw_pts_depth, forw_pts_depth_aligned;

    vector<cv::Point2f> prev_un_pts, cur_un_pts;
    vector<cv::Point2f> prev_un_pts_depth, cur_un_pts_depth;
    vector<cv::Point2f> pts_velocity;
    vector<cv::Point2f> pts_velocity_depth;
    vector<cv::Point3f> mvX3Dc1;
    vector<cv::Point3f> mvX3Dc2;
    vector<int> ids;
    vector<int> track_cnt;
    vector<int> ids_depth;
    vector<int> track_cnt_depth;
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    map<int, cv::Point2f> cur_un_pts_map_depth;
    map<int, cv::Point2f> prev_un_pts_map_depth;
    camodocal::CameraPtr m_camera, m_depth_camera;
    double cur_time;
    double prev_time;

    static int n_id;
};
