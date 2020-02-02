// Std headers
#include <iostream>
#include <vector>
#include <chrono>

// OpenCV headers
#include <opencv2/core/core.hpp>

// Ceres headers
#include <ceres/ceres.h>

// The model of the cost function
// y = exp(ax^2 + bx + c)
struct CURVE_FITTING_COST
{
    CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

    // Calculate error
    template<typename T>
    bool operator() (const T *const abc, T *residual) const
    {
         residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
        return true;
    }

    const double _x, _y;
};

int main(int argc, char **argv)
{
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

    double abc[3] = {ae, be, ce};
    
    // Construct the least square method.
    ceres::Problem problem;
    for (int i = 0; i < N; ++i)
    {
        problem.AddResidualBlock(
            // Use AutoDiffCostFunction which computes the derivative of the cost with respect to  
            // the parameters (a.k.a. the jacobian) using an autodifferentiation framework.  
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
                new CURVE_FITTING_COST(x_data[i], y_data[i])
            ),
            // No loss function
            nullptr,
            // Parameter that be estimated
            abc
        );
    }

    // Construct the solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "solve  time cost = " << time_used.count() << " seconds." << std::endl;

    // Output ceres summary
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "estimated a, b, c = ";
    for (auto a : abc) std::cout << a << " ";
    std::cout << std::endl;

    return 0;
}