// Std headers
#include <iostream>
#include <chrono>
#include <vector>

// OpenCV headers
#include <opencv2/opencv.hpp>

// Eigen headers
#include <Eigen/Core>
#include <Eigen/Dense>

int main(int argc, char **argv)
{
    // Given function y = exp(ax^2 + bx + c) + w, with the given x, y value
    // and gaussian noise, using gauss-newton method to find the coefficients
    // a, b, c that match given x, y value.

    // The real coefficients
    double ar = 1.0, br = 2.0, cr = 1.0;
    // The initial value of coefficients that used by gauss-newton method
    double ae = 2.0, be = -1.0, ce = 5.0;
    // The amount of data
    int N = 100;
    // The standard deviation of gaussian noise
    double w_sigma = 1.0;
    double inv_sigma = 1.0 / w_sigma;

    // Random generator from OpenCV
    cv::RNG rng;

    // Generate data
    std::vector<double> x_data, y_data;
    for (int i = 0; i < N; ++i)
    {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(std::exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    // Start iterating through gauss-newton method
    int iterations = 100;
    double cost = 0, lastCost = 0;

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
    {
        // Hessian = J * W^-1 * J^T
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        // Bias
        Eigen::Vector3d b = Eigen::Vector3d::Zero();
        cost = 0;

        for (int i = 0; i < N; ++i)
        {
            double xi = x_data[i], yi = y_data[i];
            double error = yi - std::exp(ae * xi * xi + be * xi + ce);

            // Calculate jacobian matrix
            Eigen::Vector3d J;
            // Error differentiate with respect to ae, de/ da
            J[0] = -xi * xi * std::exp(ae * xi * xi + be * xi + ce);
            // Error differentiate with respect to be, de/db
            J[1] = -xi * std::exp(ae * xi * xi + be * xi + ce);
            // Error differentiate with respect to ae, de/dc
            J[2] = -std::exp(ae * xi * xi + be * xi + ce);

            H += inv_sigma * inv_sigma * J * J.transpose();
            b += -inv_sigma * error * J;

            cost += error * error;
        }

        // Calculate Hx = b
        Eigen::Vector3d dx = H.ldlt().solve(b);
        if (std::isnan(dx[0]))
        {
            std::cout << "result is nan" << std::endl;
            break;
        }

        if (iter > 0 && cost >= lastCost)
        {
            std::cout << "cost: " << cost << ">= last cost: " << lastCost << ", break." << std::endl;
            break;
        }

        ae += dx[0];
        be += dx[1];
        ce += dx[2];
        lastCost = cost;

        std::cout << "total cost: " << cost << ", \t\tupdate: "<< dx.transpose() << 
            "\t\testimated params: " << ae << "," << be << "," << ce << std::endl;
    }
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    std::cout << "solve time cost = " << time_used.count() << " seconds" << std::endl;
    std::cout << "estimated a, b, c = " << ae << ", " << be << ", " << ce << std::endl;
    return 0;
}