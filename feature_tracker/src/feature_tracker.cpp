#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

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

inline void computeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C)
{
    cv::reduce(P,C,1,CV_REDUCE_SUM);
    C = C/P.cols;

    for(int i=0; i<P.cols; i++)
    {
        Pr.col(i) = P.col(i) - C;
    }
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
    vector<pair<int, pair<pair<cv::Point2f, int>, cv::Point3f>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(make_pair(forw_pts[i], ids[i]), mvX3Dc2[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<pair<cv::Point2f, int>, cv::Point3f>> &a, const pair<int, pair<pair<cv::Point2f, int>, cv::Point3f>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();
    mvX3Dc2.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first.first) == 255)
        {
            forw_pts.push_back(it.second.first.first);
            ids.push_back(it.second.first.second);
            track_cnt.push_back(it.first);
            mvX3Dc2.push_back(it.second.second);
            cv::circle(mask, it.second.first.first, MIN_DIST, 0, -1);
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
        Eigen::Vector2d a(p.x, p.y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        int ff = (int)forw_depth.at<unsigned short>(floor(p.y), floor(p.x));
        int cf = (int)forw_depth.at<unsigned short>(floor(p.y), ceil(p.x));
        int fc = (int)forw_depth.at<unsigned short>(ceil(p.y), floor(p.x));
        int cc = (int)forw_depth.at<unsigned short>(ceil(p.y), ceil(p.x));
        float count = ((float)(ff > 0) + (float)(cf > 0) + (float)(fc > 0) + (float)(cc > 0));
        float depth_val = count > 0 ? (ff + cf + fc + cc) / count / 1000:0;
        mvX3Dc2.push_back(cv::Point3f (b.x() / b.z() * depth_val, b.y() / b.z() * depth_val, depth_val));
    }
}

void FeatureTracker::readImage(const cv::Mat &_img, const cv::Mat &_depth, double _cur_time)
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
    }
    else
    {
        forw_img = img;
        forw_depth = _depth;
    }

    forw_pts.clear();

    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        //光流跟踪点
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

    for (auto &n : track_cnt)
        n++;

    if (PUB_THIS_FRAME)
    {
        //对prev_pts和forw_pts做ransac剔除outlier.
        //rejectWithF();
        rejectWithSim3();
        //rejectWithPnP();
        TicToc t_m;
        //有点类似non-max suppression
        ROS_DEBUG("set mask begins");
        setMask();
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
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    prev_img = cur_img;
    prev_depth = cur_depth;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_depth = forw_depth;
    cur_pts = forw_pts;
    undistortedPoints();
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
        // in case of bad depth, mvX3Dc2 here may have many zeros
        reduceVector(mvX3Dc2, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

void FeatureTracker::rejectWithSim3()
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

    double nonzero_depth_count = 0;
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
        if (status[i])
            ++nonzero_depth_count;
    }
    // less than 30% have depth, we got a bad depth map.
    // use rejectWithF instead.
    //if (1)  //uncomment this if don't want to use sim3, then it will always call rejectWithF
    if (nonzero_depth_count/forw_pts.size() < 0.3)
    {
        rejectWithF();
        //ROS_ERROR("Bad depth, call rejectWithF()");
        return;
    }

    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(cur_un_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
    reduceVector(mvX3Dc1, status);
    //after here, mvX3Dc2 contains no zeros.
    reduceVector(mvX3Dc2, status);
    //ROS_ERROR("Points before depth filter: %d      After depth filter: %d", N, forw_pts.size());
    N = forw_pts.size();
    if (forw_pts.size() < 8)
    {
        ROS_ERROR("No enough depth");
        return;
    }
    // Set rand seed
    srand(time(NULL));
    // iterate at most 200 times, if Inliers > 0.7*all, stop
    int mRansacMaxIts = 200;
    int mRansacMinInliers = nonzero_depth_count * 0.8;

    // mvAllIndices[i] = i
    vector<size_t> mvAllIndices(N);
    vector<size_t> vAvailableIndices;

    for (size_t i = 0; i < N; i++)
        // Setup this to perform random
        mvAllIndices[i] = i;
    vector<uchar> vbInliers(N, 0);
    vector<uchar> vbBestInliers(N, 0);


    cv::Mat P3Dc1i(3,3,CV_32F);
    cv::Mat P3Dc2i(3,3,CV_32F);

    int mnBestInliers = 0;
    int mnIterations = 0;
    while(mnIterations<mRansacMaxIts)
    {

        vAvailableIndices = mvAllIndices;
        // Get min set of points
        // 步骤1：任意取三组点算Sim矩阵
        for(short i = 0; i < 3; ++i)
        {
            unsigned long randi = rand()%(vAvailableIndices.size());
            unsigned long idx = vAvailableIndices[randi];

            // P3Dc1i和P3Dc2i中点的排列顺序：
            // x1 x2 x3 ...
            // y1 y2 y3 ...
            // z1 z2 z3 ...
            cv::Mat(mvX3Dc1[idx]).col(0).copyTo(P3Dc1i.col(i));
            cv::Mat(mvX3Dc2[idx]).col(0).copyTo(P3Dc2i.col(i));

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
        cv::Mat mR12i(3,3,CV_32F);
        cv::Mat mt12i(3,1,CV_32F);
        // 步骤2：根据两组匹配的3D点，计算之间的Sim3变换
        // computeSim3 return R and t from forw to cur
        if (computeSim3(P3Dc1i, P3Dc2i, mR12i, mt12i))
            mnIterations++;// 总的迭代次数
        else
            continue;

        // 步骤3：通过投影误差进行inlier检测
        //checkInliers(const vector<cv::Point3f> &mvX3Dc2, const vector<cv::Point2f> &mvP1im1, cv::Mat mR12i, cv::Mat mt12i, vector<uchar> &mvbInliersi)
        int mnInliersi = checkInliers(mR12i, mt12i, vbInliers);
        if(mnInliersi>=mnBestInliers)
        {
            vbBestInliers = vbInliers;
            mnBestInliers = mnInliersi;

            if(mnInliersi>mRansacMinInliers)// 只要计算得到一次合格的Sim变换，就直接返回
                break;
        }
    }
    ROS_WARN("mnBestInliers: %d", mnBestInliers);
    reduceVector(prev_pts, vbBestInliers);
    reduceVector(cur_pts, vbBestInliers);
    reduceVector(forw_pts, vbBestInliers);
    reduceVector(cur_un_pts, vbBestInliers);
    reduceVector(ids, vbBestInliers);
    reduceVector(track_cnt, vbBestInliers);
}

bool FeatureTracker::computeSim3(cv::Mat &P1, cv::Mat &P2, cv::Mat &mR12i, cv::Mat &mt12i)
{
    // Custom implementation of:
    // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

    // Step 1: Centroid and relative coordinates
    cv::Mat Pr1(P1.size(),P1.type()); // Relative coordinates to centroid (set 1)
    cv::Mat Pr2(P2.size(),P2.type()); // Relative coordinates to centroid (set 2)
    cv::Mat O1(3,1,Pr1.type()); // Centroid of P1
    cv::Mat O2(3,1,Pr2.type()); // Centroid of P2

    computeCentroid(P1,Pr1,O1);
    computeCentroid(P2,Pr2,O2);

    // Step 2: Compute M matrix

    cv::Mat M = Pr2*Pr1.t();

    // Step 3: Compute N matrix

    double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

    cv::Mat N(4,4,P1.type());

    N11 = M.at<float>(0,0)+M.at<float>(1,1)+M.at<float>(2,2);
    N12 = M.at<float>(1,2)-M.at<float>(2,1);
    N13 = M.at<float>(2,0)-M.at<float>(0,2);
    N14 = M.at<float>(0,1)-M.at<float>(1,0);
    N22 = M.at<float>(0,0)-M.at<float>(1,1)-M.at<float>(2,2);
    N23 = M.at<float>(0,1)+M.at<float>(1,0);
    N24 = M.at<float>(2,0)+M.at<float>(0,2);
    N33 = -M.at<float>(0,0)+M.at<float>(1,1)-M.at<float>(2,2);
    N34 = M.at<float>(1,2)+M.at<float>(2,1);
    N44 = -M.at<float>(0,0)-M.at<float>(1,1)+M.at<float>(2,2);

    N = (cv::Mat_<float>(4,4) << N11, N12, N13, N14,
            N12, N22, N23, N24,
            N13, N23, N33, N34,
            N14, N24, N34, N44);


    // Step 4: Eigenvector of the highest eigenvalue

    cv::Mat eval, evec;

    cv::eigen(N,eval,evec); //evec[0] is the quaternion of the desired rotation

    cv::Mat vec(1,3,evec.type());
    (evec.row(0).colRange(1,4)).copyTo(vec); //extract imaginary part of the quaternion (sin*axis)

    // Rotation angle. sin is the norm of the imaginary part, cos is the real part
    double ang=atan2(norm(vec),evec.at<float>(0,0));

    vec = 2*ang*vec/norm(vec); //Angle-axis representation. quaternion angle is the half

    mR12i.create(3,3,P1.type());

    cv::Rodrigues(vec,mR12i); // computes the rotation matrix from angle-axis

    // Step 5: Rotate set 2

    cv::Mat P3 = mR12i*Pr2;

    // Step 6: Scale

    double ms12i = 0;
    {
        double nom = Pr1.dot(P3);
        cv::Mat aux_P3(P3.size(),P3.type());
        aux_P3=P3;
        cv::pow(P3,2,aux_P3);
        double den = 0;

        for(int i=0; i<aux_P3.rows; i++)
        {
            for(int j=0; j<aux_P3.cols; j++)
            {
                den+=aux_P3.at<float>(i,j);
            }
        }

        ms12i = nom/den;
    }
    // Step 6: Scale=1 pass
    if (abs(ms12i-1)>0.1)
        return false;
    // Step 7: Translation

    mt12i.create(1,3,P1.type());
    mt12i = O1 - ms12i*mR12i*O2;

    return true;
}


int FeatureTracker::checkInliers(cv::Mat mR12i, cv::Mat mt12i, vector<uchar> &mvbInliersi)
{
    vector<cv::Point2f> vP2im1;
    project(vP2im1, mR12i, mt12i);
    //m_camera->projectPoints(mvX3Dc2, mR12i, mt12i, vP2im1);
    double mvnMaxError = 10;
    int mnInliersi=0;

    for(size_t i=0; i<cur_pts.size(); i++)
    {
        cv::Point2f dist = cur_pts[i]-vP2im1[i];

        const float err = dist.dot(dist);
        //cout<<"ERROR: "<<err<<endl;
        if(err<mvnMaxError)
        {
            mvbInliersi[i]=1;
            mnInliersi++;
        }
    }
    return mnInliersi;
}

void FeatureTracker::project(vector<cv::Point2f> &vP2D, cv::Mat mR12i, cv::Mat mt12i)
{
    const float &fx = 6.1659e+02;
    const float &fy = 6.1668e+02;
    const float &cx = 3.2422e+02;
    const float &cy = 2.3943e+02;

    vP2D.clear();
    vP2D.reserve(mvX3Dc2.size());

    for (size_t i=0, iend=mvX3Dc2.size(); i<iend; i++)
    {
        cv::Mat P3Dc = mR12i*cv::Mat(mvX3Dc2[i])+mt12i;
        const float invz = 1/(P3Dc.at<float>(2));
        const float x = P3Dc.at<float>(0)*invz;
        const float y = P3Dc.at<float>(1)*invz;

        vP2D.push_back(cv::Point2f(fx*x+cx, fy*y+cy));
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

    double nonzero_depth_count = 0;
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
        if (status[i])
            ++nonzero_depth_count;
    }
    // less than 30% have depth, we got a bad depth map.
    // use rejectWithF instead.
    if (nonzero_depth_count/forw_pts.size() < 0.3)
    {
        rejectWithF();
        ROS_ERROR("Bad depth, call rejectWithF()");
        return;
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

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
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
