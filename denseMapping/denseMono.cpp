#include <bits/stdc++.h>
#include <boost/timer.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// parameters
const int boarder = 20;
const int width = 640;
const int height = 480;
const double fx = 481.2f;
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 3;
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1);
const double min_cov = 0.1;
const double max_cov = 10;

bool readDatasetFiles(
    const std::string &path,
    std::vector<std::string> &color_image_files,
    std::vector<Sophus::SE3d> &poses,
    cv::Mat &ref_depth
)
{
    std::ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin) return false;

    while (!fin.eof())
    {
        std::string image;
        fin >> image;
        double data[7];
        for (double &d : data) fin >> d;

        color_image_files.push_back(path + std::string("/images/") + image);
        poses.push_back(
            Sophus::SE3d(Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                         Eigen::Vector3d(data[0], data[1], data[2]))
        );
        if (!fin.good()) break;
    }
    fin.close();

    fin.open(path + "/depthmaps/scene_000.depth");
    ref_depth = cv::Mat(height, width, CV_64F);
    if (!fin) return false;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            double depth = 0;
            fin >> depth;
            ref_depth.ptr<double>(y)[x] = depth / 100.0;
        }
    }
    return true;
}

double getBilinearInterpolatedValue(const cv::Mat &img, const Eigen::Vector2d &pt)
{
    uchar *d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
    double xx = pt(0, 0) - floor(pt(0, 0));
    double yy = pt(1, 0) - floor(pt(1, 0));
    return ((1 - xx) * (1 - yy) * double(d[0]) +
            xx * (1 - yy) * double(d[1]) +
            (1 - xx) * yy * double(d[img.step]) +
            xx * yy * double(d[img.step + 1])) / 255.0;
}

void plotDepth(const cv::Mat &depth_truth, const cv::Mat &depth_estimate)
{
    cv::imshow("depth_truth", depth_truth * 0.4);
    cv::imshow("depth_estimate", depth_estimate * 0.4);
    cv::imshow("depth_error", depth_truth - depth_estimate);
    cv::waitKey(1);
}

Eigen::Vector3d px2cam(const Eigen::Vector2d px) {
    return Eigen::Vector3d(
        (px(0, 0) - cx) / fx,
        (px(1, 0) - cy) / fy,
        1
    );
}

Eigen::Vector2d cam2px(const Eigen::Vector3d p_cam) {
    return Eigen::Vector2d(
        p_cam(0, 0) * fx / p_cam(2, 0) + cx,
        p_cam(1, 0) * fy / p_cam(2, 0) + cy
    );
}

bool inside(const Eigen::Vector2d &pt) {
    return pt(0, 0) >= boarder && pt(1, 0) >= boarder
           && pt(0, 0) + boarder < width && pt(1, 0) + boarder <= height;
}

void showEpipolarMatch(const cv::Mat &ref, const cv::Mat &curr, const Eigen::Vector2d &px_ref, const Eigen::Vector2d &px_curr)
{
    cv::Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, CV_GRAY2BGR);
    cv::cvtColor(curr, curr_show, CV_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 0, 250), 2);
    cv::circle(curr_show, cv::Point2f(px_curr(0, 0), px_curr(1, 0)), 5, cv::Scalar(0, 0, 250), 2);

    cv::imshow("ref", ref_show);
    cv::imshow("curr", curr_show);
    cv::waitKey(1);
}

void showEpipolarLine(const cv::Mat &ref, const cv::Mat &curr, const Eigen::Vector2d &px_ref, const Eigen::Vector2d &px_min_curr,
                      const Eigen::Vector2d &px_max_curr)
{
    cv::Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, CV_GRAY2BGR);
    cv::cvtColor(curr, curr_show, CV_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::line(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)),
             cv::Scalar(0, 255, 0), 1);

    cv::imshow("ref", ref_show);
    cv::imshow("curr", curr_show);
    cv::waitKey(1);
}

void evaludateDepth(const cv::Mat &depth_truth, const cv::Mat &depth_estimate)
{
    double ave_depth_error = 0;
    double ave_depth_error_sq = 0;
    int cnt_depth_data = 0;
    for (int y = boarder; y < depth_truth.rows - boarder; y++)
        for (int x = boarder; x < depth_truth.cols - boarder; x++) 
        {
            double error = depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x];
            ave_depth_error += error;
            ave_depth_error_sq += error * error;
            cnt_depth_data++;
        }
    ave_depth_error /= cnt_depth_data;
    ave_depth_error_sq /= cnt_depth_data;

    std::cout << "Average squared error = " << ave_depth_error_sq << ", average error: " << ave_depth_error << std::endl;
}

double NCC(
    const cv::Mat &ref,
    const cv::Mat &curr,
    const Eigen::Vector2d &pt_ref,
    const Eigen::Vector2d &pt_curr
)
{
    double mean_ref = 0, mean_curr = 0;
    std::vector<double> values_ref, values_curr;
    for (int x = -ncc_window_size; x <= ncc_window_size; ++x)
    {
        for (int y = -ncc_window_size; y <= ncc_window_size; ++y)
        {
            double value_ref = double(ref.ptr<uchar>(int(y + pt_ref(1,0)))[int(x + pt_ref(0, 0))]) / 255.0;
            mean_ref += value_ref;

            double value_curr = getBilinearInterpolatedValue(curr, pt_curr + Eigen::Vector2d(x, y));
            mean_curr += value_curr;

            values_ref.push_back(value_ref);
            values_curr.push_back(value_curr);
        }
    }
    mean_ref /= ncc_area;
    mean_curr /= ncc_area;

    double numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (int i = 0; i < values_ref.size(); ++i)
    {
        double n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr);
        numerator += n;
        demoniator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
        demoniator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr);
    }

    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}

bool epipolarSearch(
    const cv::Mat &ref,
    const cv::Mat &curr,
    const Sophus::SE3d &T_C_R,
    const Eigen::Vector2d &pt_ref,
    const double &depth_mu,
    const double &depth_cov,
    Eigen::Vector2d &pt_curr,
    Eigen::Vector2d &epipolar_direction
)
{
    Eigen::Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    Eigen::Vector3d P_ref = f_ref * depth_mu;

    Eigen::Vector2d px_mean_curr = cam2px(T_C_R * P_ref);
    double d_min = depth_mu - 3 * depth_cov, d_max = depth_mu + 3 * depth_cov;
    if (d_min < 0.1) d_min = 0.1;
    Eigen::Vector2d px_min_curr = cam2px(T_C_R * (f_ref * d_min));
    Eigen::Vector2d px_max_curr = cam2px(T_C_R * (f_ref * d_max));

    Eigen::Vector2d epipolar_line = px_max_curr - px_min_curr;
    epipolar_direction = epipolar_line;
    epipolar_direction.normalize();
    double half_length = 0.5 * epipolar_line.norm();
    if (half_length > 100) half_length = 100;

    // showEpipolarLine(ref, curr, pt_ref, px_min_curr, px_max_curr);
    double best_ncc = -1.0;
    Eigen::Vector2d best_px_curr;
    for (double l = -half_length; l <= half_length; l += 0.7)
    {
        Eigen::Vector2d px_curr = px_mean_curr + l * epipolar_direction;
        if (!inside(px_curr))
            continue;
        double ncc = NCC(ref, curr, pt_ref, px_curr);
        if (ncc > best_ncc)
        {
            best_ncc = ncc;
            best_px_curr = px_curr;
        }
    }
    if (best_ncc < 0.85f)
        return false;
    pt_curr = best_px_curr;
    return true;
}

bool updateDepthFilter(
    const Eigen::Vector2d &pt_ref,
    const Eigen::Vector2d &pt_curr,
    const Sophus::SE3d &T_C_R,
    const Eigen::Vector2d &epipolar_direction,
    cv::Mat &depth,
    cv::Mat &depth_cov2
)
{
    Sophus::SE3d T_R_C = T_C_R.inverse();
    Eigen::Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    Eigen::Vector3d f_curr = px2cam(pt_curr);
    f_curr.normalize();
    
    // 方程
    // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC
    // f2 = R_RC * f_cur
    // 转化成下面这个矩阵方程组
    // => [ f_ref^T f_ref, -f_ref^T f2 ] [d_ref]   [f_ref^T t]
    //    [ f_cur^T f_ref, -f2^T f2    ] [d_cur] = [f2^T t   ]
    Eigen::Vector3d t = T_R_C.translation();
    Eigen::Vector3d f2 = T_R_C.so3() * f_curr;
    Eigen::Vector2d b = Eigen::Vector2d(t.dot(f_ref), t.dot(f2));
    Eigen::Matrix2d A;
    A(0, 0) = f_ref.dot(f_ref);
    A(0, 1) = -f_ref.dot(f2);
    A(1, 0) = -A(0, 1);
    A(1, 1) = -f2.dot(f2);
    Eigen::Vector2d ans = A.inverse() * b;
    Eigen::Vector3d xm = ans[0] * f_ref;           // ref 侧的结果
    Eigen::Vector3d xn = t + ans[1] * f2;          // cur 结果
    Eigen::Vector3d p_esti = (xm + xn) / 2.0;      // P的位置，取两者的平均
    double depth_estimation = p_esti.norm();   // 深度值

    // 计算不确定性（以一个像素为误差）
    Eigen::Vector3d p = f_ref * depth_estimation;
    Eigen::Vector3d a = p - t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f_ref.dot(t) / t_norm);
    double beta = acos(-a.dot(t) / (a_norm * t_norm));
    Eigen::Vector3d f_curr_prime = px2cam(pt_curr + epipolar_direction);
    f_curr_prime.normalize();
    double beta_prime = acos(f_curr_prime.dot(-t) / t_norm);
    double gamma = M_PI - alpha - beta_prime;
    double p_prime = t_norm * sin(beta_prime) / sin(gamma);
    double d_cov = p_prime - depth_estimation;
    double d_cov2 = d_cov * d_cov;

    // 高斯融合
    double mu = depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];
    double sigma2 = depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];

    double mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
    double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2);

    depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = mu_fuse;
    depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = sigma_fuse2;

    return true;
}

bool update(
    const cv::Mat &ref,
    const cv::Mat &curr,
    const Sophus::SE3d &T_C_R,
    cv::Mat &depth,
    cv::Mat &depth_cov2
)
{
    for (int x = boarder; x < width - boarder; ++x)
    {
        for (int y = boarder; y < height - boarder; y++)
        {
            if (depth_cov2.ptr<double>(y)[x] < min_cov || depth_cov2.ptr<double>(y)[x] > max_cov)
                continue;
            Eigen::Vector2d pt_curr;
            Eigen::Vector2d epipolar_direction;
            bool ret = epipolarSearch(
                ref, 
                curr,
                T_C_R,
                Eigen::Vector2d(x, y),
                depth.ptr<double>(y)[x],
                std::sqrt(depth_cov2.ptr<double>(y)[x]),
                pt_curr,
                epipolar_direction
            );

            if (ret == false)
                continue;

            // showEpipolarMatch(ref, curr, Eigen::Vector2d(x, y), pt_curr);
            updateDepthFilter(Eigen::Vector2d(x, y), pt_curr, T_C_R, epipolar_direction, depth, depth_cov2);
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: denseMono path_to_test_dataset" << std::endl;
        return -1;
    }

    std::vector<std::string> color_image_files;
    std::vector<Sophus::SE3d> poses_TWC;
    cv::Mat ref_depth;
    bool ret = readDatasetFiles(argv[1], color_image_files, poses_TWC, ref_depth);
    if (ret == false)
    {
        std::cout << "Read image files failed" << std::endl;
        return -1;
    }

    cv::Mat ref = cv::imread(color_image_files[0], 0);
    Sophus::SE3d pose_ref_TWC = poses_TWC[0];
    double init_depth = 3.0;
    double init_cov2 = 3.0;
    cv::Mat depth(height, width, CV_64F, init_depth);
    cv::Mat depth_cov2(height, width, CV_64F, init_cov2);

    for (int index = 1; index < color_image_files.size(); ++index)
    {
        std::cout << "*** loop " << index << " ***" << std::endl;
        cv::Mat curr = cv::imread(color_image_files[index], 0);
        if (curr.data == nullptr)
            continue;
        Sophus::SE3d pose_curr_TWC = poses_TWC[index];
        Sophus::SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC;
        update(ref, curr, pose_T_C_R, depth, depth_cov2);
        evaludateDepth(ref_depth, depth);
        plotDepth(ref_depth, depth);
        cv::imshow("image", curr);
        cv::waitKey(1);
    }

    std::cout << "estimation returns, saving depth map ..." << std::endl;
    cv::imwrite("depth.png", depth);
    std::cout << "done." << std::endl;

    return 0;
}