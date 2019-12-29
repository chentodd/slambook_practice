// Std headers
#include <iostream>
#include <cmath>

// Eigen headers
#include <Eigen/Core>
#include <Eigen/Geometry>

// Sophus headers
#include <sophus/se3.hpp>

int main(int argc, char **argv)
{
    // Rotate 90 degree with z axis
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();

    // Create quaternion from rotation matrix
    Eigen::Quaterniond q(R);

    // Create SO3 from rotation matrix and quaternion
    Sophus::SO3d SO3_R(R);
    Sophus::SO3d SO3_q(q);
    std::cout << "SO(3) from matrix: \n" << SO3_R.matrix() << std::endl;
    std::cout << "SO(3) from quaternion: \n" << SO3_q.matrix() << std::endl;

    // Get se3 using ln
    Eigen::Vector3d so3 = SO3_R.log();
    std::cout << "so3 = " << so3.transpose() << std::endl;
    // hat will transform vector to antisymmetric matrix
    std::cout << "so3 hat = \n" << Sophus::SO3d::hat(so3) << std::endl;
    // vee will transform antisymmetric matrix  to vector
    std::cout << "so3 hat vee = \n" << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose() << std::endl;

    // Update will disturbance model
    Eigen::Vector3d update_so3(1e-4, 0, 0);
    Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
    std::cout << "SO3 updated = \n" << SO3_updated.matrix() << std::endl;

    std::cout << "*************************************************************" << std::endl;
    // Using SE(3)
    Eigen::Vector3d t(1, 0, 0);
    Sophus::SE3d SE3_Rt(R, t);
    Sophus::SE3d SE3_qt(q, t);
    std::cout << "SE3 from R, t = \n" << SE3_Rt.matrix() << std::endl;
    std::cout << "SE3 from q, t = \n" << SE3_qt.matrix() << std::endl;
    // se(3) is a 6D vector
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    Vector6d se3 = SE3_Rt.log();
    std::cout << "se3 = " << se3.transpose() << std::endl;
    std::cout << "se3 hat = \n" << Sophus::SE3d::hat(se3) << std::endl;
    std::cout << "se3 hat ver = \n" << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose() << std::endl;

    Vector6d update_se3;
    update_se3.setZero();
    update_se3(0, 0) = 1e-4;
    Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
    std::cout << "SE3 updated = " << std::endl << SE3_updated.matrix() << std::endl;

    return 0;
}