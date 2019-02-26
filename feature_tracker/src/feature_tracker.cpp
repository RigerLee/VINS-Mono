#include "feature_tracker.h"

int FeatureTracker::n_id = 0;
Eigen::Matrix3d RR;
Eigen::Vector3d TT;


bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<cv::Point3f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));


    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::setMaskDepth()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));


    // prefer to keep features that are tracked for long time
    vector<pair<pair<int, cv::Point2f>, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts_depth.size(); i++)
        cnt_pts_id.push_back(make_pair(make_pair(track_cnt_depth[i], forw_pts_depth_aligned[i]), make_pair(forw_pts_depth[i], ids_depth[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<pair<int, cv::Point2f>, pair<cv::Point2f, int>> &a, const pair<pair<int, cv::Point2f>, pair<cv::Point2f, int>> &b)
    {
        return a.first.first > b.first.first;
    });

    forw_pts_depth.clear();
    forw_pts_depth_aligned.clear();
    ids_depth.clear();
    track_cnt_depth.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.first.second) == 255)
        {
            forw_pts_depth.push_back(it.second.first);
            forw_pts_depth_aligned.push_back(it.first.second);
            ids_depth.push_back(it.second.second);
            track_cnt_depth.push_back(it.first.first);
            cv::circle(mask, it.first.second, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
    for (auto &p : n_pts_depth)
    {
        forw_pts_depth.push_back(p);
        ids_depth.push_back(-1);
        track_cnt_depth.push_back(1);
    }
}

void FeatureTracker::readImage(const cv::Mat &_img, const cv::Mat &_color_depth, const cv::Mat &_depth, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;
    // too dark or too bright: histogram
    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    if (forw_img.empty())
    {
        //curr_img<--->forw_img

        prev_img = cur_img = forw_img = img;
        prev_depth = cur_depth = forw_depth = _depth;
        prev_color_depth = cur_color_depth = forw_color_depth = _color_depth;
    }
    else
    {
        forw_img = img;
        forw_depth = _depth;
        forw_color_depth = _color_depth;
    }

    forw_pts.clear();
    forw_pts_depth.clear();

    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        //剔除图像边缘的点
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);

        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }
    if (cur_pts_depth.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        cv::calcOpticalFlowPyrLK(cur_color_depth, forw_color_depth, cur_pts_depth, forw_pts_depth, status, err, cv::Size(21, 21), 3);

        //剔除图像边缘的点
        for (int i = 0; i < int(forw_pts_depth.size()); i++)
            if (status[i] && !inBorder(forw_pts_depth[i]))
                status[i] = 0;
        reduceVector(prev_pts_depth, status);
        reduceVector(cur_pts_depth, status);
        reduceVector(forw_pts_depth, status);
        reduceVector(ids_depth, status);
        reduceVector(cur_un_pts_depth, status);
        reduceVector(track_cnt_depth, status);

        //do transform for forw here
        vector<uchar> status1(forw_pts_depth.size());
        RR << 0.9999530, -0.0084620, -0.0047202, 0.0084411,  0.9999546, -0.0044289, 0.0047575,  0.0043889,  0.9999791;
        TT << 0.00034269, -0.01478, -0.00019627;
        forw_pts_depth_aligned.clear();
        for (unsigned int i = 0; i < forw_pts_depth.size(); i++)
        {
            int ff = (int)forw_depth.at<unsigned short>(floor(forw_pts_depth[i].y), floor(forw_pts_depth[i].x));
            int cf = (int)forw_depth.at<unsigned short>(floor(forw_pts_depth[i].y), ceil(forw_pts_depth[i].x));
            int fc = (int)forw_depth.at<unsigned short>(ceil(forw_pts_depth[i].y), floor(forw_pts_depth[i].x));
            int cc = (int)forw_depth.at<unsigned short>(ceil(forw_pts_depth[i].y), ceil(forw_pts_depth[i].x));
            int count = ((int)(ff > 0) + (int)(cf > 0) + (int)(fc > 0) + (int)(cc > 0));
            double avg_depth = count > 0 ? (ff + cf + fc + cc) / count:0;
            if (avg_depth == 0.0)
            {
                status1[i] = 0;
                continue;
            }
            avg_depth = avg_depth / 1000;

            Eigen::Vector3d tmp_p;
            m_depth_camera->liftProjective(Eigen::Vector2d(forw_pts_depth[i].x, forw_pts_depth[i].y), tmp_p);
            tmp_p = tmp_p * avg_depth;
            tmp_p = RR * tmp_p + TT;
            Eigen::Vector2d tmp_p_2d;
            m_camera->spaceToPlane(tmp_p, tmp_p_2d);
            cv::Point2f temp_cvp(tmp_p_2d(0, 0), tmp_p_2d(1, 0));
            if (!inBorder(temp_cvp))
            {
                status1[i] = 0;
                continue;
            }
            forw_pts_depth_aligned.push_back(temp_cvp);
            status1[i] = 1;
        }
        reduceVector(prev_pts_depth, status1);
        reduceVector(cur_pts_depth, status1);
        reduceVector(forw_pts_depth, status1);
        reduceVector(ids_depth, status1);
        reduceVector(cur_un_pts_depth, status);
        reduceVector(track_cnt_depth, status1);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt)
        n++;
    for (auto &n : track_cnt_depth)
        n++;
    if (PUB_THIS_FRAME)
    {
        //对prev_pts和forw_pts做ransac剔除outlier.
        rejectWithF();
        rejectWithFDepth();
        ROS_DEBUG("set mask begins");
        //rejectWithSim3();
        //rejectWithPnP();
        TicToc t_m;
        //有点类似non-max suppression(Sort and surpress)
        setMask();
        setMaskDepth();
        ROS_DEBUG("set mask costs %fms", t_m.toc());




        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            //提取新的角点n_pts, 使数量达到MAX_CNT, 通过addPoints()函数push到forw_pts中, id初始化-1,track_cnt初始化为1.
            //初始cur_pts.size = 0, 所有pts来自Shi-Tomasi角点检测(改进Harris)
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();

        int n_max_cnt_depth = MAX_CNT - static_cast<int>(forw_pts_depth.size());
        if (n_max_cnt_depth > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_color_depth.size())
                cout << "wrong size " << endl;
            cv::goodFeaturesToTrack(forw_color_depth, n_pts_depth, MAX_CNT - forw_pts_depth.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    prev_img = cur_img;
    prev_depth = cur_depth;
    prev_pts = cur_pts;
    prev_pts_depth = cur_pts_depth;
    prev_color_depth = cur_color_depth;
    prev_un_pts = cur_un_pts;
    prev_un_pts_depth = cur_un_pts_depth;
    cur_img = forw_img;
    cur_depth = forw_depth;
    cur_pts = forw_pts;
    cur_pts_depth = forw_pts_depth;
    cur_color_depth = forw_color_depth;
    undistortedPoints();
    undistortedPointsDepth();
    prev_time = cur_time;
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}
void FeatureTracker::rejectWithFDepth()
{
    if (forw_pts_depth.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts_depth.size()), un_forw_pts(forw_pts_depth.size());
        for (unsigned int i = 0; i < cur_pts_depth.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts_depth[i].x, cur_pts_depth[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts_depth[i].x, forw_pts_depth[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts_depth, status);
        reduceVector(cur_pts_depth, status);
        reduceVector(forw_pts_depth, status);
        reduceVector(ids_depth, status);
        reduceVector(track_cnt_depth, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts_depth.size(), 1.0 * forw_pts_depth.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

void FeatureTracker::rejectWithPnP()
{
    if (forw_pts.size() < 8)
    {
        ROS_ERROR("No enough points");
        return;
    }
    unsigned long N = forw_pts.size();
    vector<uchar> status(N, 1);
    mvX3Dc1.resize(N);
    mvX3Dc2.resize(N);
    //后面考虑是否从forw project到cur更合适?
    for (size_t i = 0; i < N; i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        int ff = (int)cur_depth.at<unsigned short>(floor(cur_pts[i].y), floor(cur_pts[i].x));
        int cf = (int)cur_depth.at<unsigned short>(floor(cur_pts[i].y), ceil(cur_pts[i].x));
        int fc = (int)cur_depth.at<unsigned short>(ceil(cur_pts[i].y), floor(cur_pts[i].x));
        int cc = (int)cur_depth.at<unsigned short>(ceil(cur_pts[i].y), ceil(cur_pts[i].x));
        float count = ((float)(ff > 0) + (float)(cf > 0) + (float)(fc > 0) + (float)(cc > 0));
        float depth_val = count > 0 ? (ff + cf + fc + cc) / count / 1000:0;
        mvX3Dc1[i] = cv::Point3f (b.x() / b.z() * depth_val, b.y() / b.z() * depth_val, depth_val);
        // Skip points with depth=0
        if (depth_val == 0.0)
            status[i] = 0;

        a = Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y);
        m_camera->liftProjective(a, b);
        ff = (int)forw_depth.at<unsigned short>(floor(forw_pts[i].y), floor(forw_pts[i].x));
        cf = (int)forw_depth.at<unsigned short>(floor(forw_pts[i].y), ceil(forw_pts[i].x));
        fc = (int)forw_depth.at<unsigned short>(ceil(forw_pts[i].y), floor(forw_pts[i].x));
        cc = (int)forw_depth.at<unsigned short>(ceil(forw_pts[i].y), ceil(forw_pts[i].x));
        count = ((float)(ff > 0) + (float)(cf > 0) + (float)(fc > 0) + (float)(cc > 0));
        depth_val = count > 0 ? (ff + cf + fc + cc) / count / 1000:0;
        mvX3Dc2[i] = cv::Point3f (b.x() / b.z() * depth_val, b.y() / b.z() * depth_val, depth_val);
        // Skip points with depth=0
        if (depth_val == 0.0)
            status[i] = 0;

    }

    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(cur_un_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
    reduceVector(mvX3Dc1, status);
    //ROS_ERROR("Points before depth filter: %d      After depth filter: %d", N, forw_pts.size());
    N = forw_pts.size();
    if (forw_pts.size() < 8)
    {
        ROS_ERROR("No enough depth");
        return;
    }

    cv::Mat intrinsics, distCoeffs;
    //intrinsics = (cv::Mat_<double>(3,3) << FOCAL_LENGTH, 0, COL / 2.0, 0, FOCAL_LENGTH, ROW / 2.0, 0, 0, 1);
    intrinsics = (cv::Mat_<double>(3,3) << 6.1659e+02, 0, 3.2422e+02, 0, 6.1668e+02, 2.3943e+02, 0, 0, 1);
    distCoeffs = (cv::Mat_<double>(1,4) << 1.25323e-01, -2.51452e-01, 7.12e-04, 6.217e-03);

    cv::Mat rvec, tvec;
    vector<int> status1;

    if (!cv::solvePnPRansac(mvX3Dc1, forw_pts, intrinsics, distCoeffs, rvec, tvec, false, 200, 1, 0.99, status1, cv::SOLVEPNP_ITERATIVE))
        return;
    ROS_ERROR("Points after PnP: %lu", status1.size());

    //cv::Mat R_test;
    //cv::Rodrigues(rvec, R_test);
    //cout<<"R:"<<R_test<<endl;
    //cout<<"t:"<<tvec<<endl;


    vector<u_char> status2(N, 0);
    for (int i = 0; i < status1.size(); i++)
        status2[status1[i]] = 1;

    reduceVector(prev_pts, status2);
    reduceVector(cur_pts, status2);
    reduceVector(forw_pts, status2);
    reduceVector(cur_un_pts, status2);
    reduceVector(ids, status2);
    reduceVector(track_cnt, status2);
    cout<<forw_pts.size()<<endl;

}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}
bool FeatureTracker::updateIDDepth(unsigned int i)
{
    if (i < ids_depth.size())
    {
        if (ids_depth[i] == -1)
            ids_depth[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file, const string &depth_calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
    m_depth_camera = CameraFactory::instance()->generateCameraFromYamlFile(depth_calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        //https://github.com/HKUST-Aerial-Robotics/VINS-Mono/blob/0d280936e441ebb782bf8855d86e13999a22da63/camera_model/src/camera_models/PinholeCamera.cc
        //brief Lifts a point from the image plane to its projective ray
        m_camera->liftProjective(a, b);
        // 特征点在相机坐标系的归一化坐标
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}

void FeatureTracker::undistortedPointsDepth()
{
    cur_un_pts_depth.clear();
    cur_un_pts_map_depth.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < forw_pts_depth_aligned.size(); i++)
    {
        Eigen::Vector2d a(forw_pts_depth_aligned[i].x, forw_pts_depth_aligned[i].y);
        Eigen::Vector3d b;
        //https://github.com/HKUST-Aerial-Robotics/VINS-Mono/blob/0d280936e441ebb782bf8855d86e13999a22da63/camera_model/src/camera_models/PinholeCamera.cc
        //brief Lifts a point from the image plane to its projective ray
        m_camera->liftProjective(a, b);
        // 特征点在相机坐标系的归一化坐标
        cur_un_pts_depth.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map_depth.insert(make_pair(ids_depth[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map_depth.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity_depth.clear();
        for (unsigned int i = 0; i < cur_un_pts_depth.size(); i++)
        {
            if (ids_depth[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map_depth.find(ids[i]);
                if (it != prev_un_pts_map_depth.end())
                {
                    double v_x = (cur_un_pts_depth[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts_depth[i].y - it->second.y) / dt;
                    pts_velocity_depth.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity_depth.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity_depth.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < forw_pts_depth_aligned.size(); i++)
        {
            pts_velocity_depth.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
    prev_un_pts_map_depth = cur_un_pts_map_depth;
}
