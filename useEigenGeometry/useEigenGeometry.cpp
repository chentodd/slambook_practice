// Std headers
#include <iostream>
#include <cmath>

// Eigen headers
#include <Eigen/Core>
#include <Eigen/Geometry>

int main(int argc, char **argv)
{
    // Create rotation matrix
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();

    // Create rotation vector, rotate with z-axis with 45deg
    Eigen::AngleAxisd rotation_vector(M_PI / 4, Eigen::Vector3d(0, 0, 1));

    // Convert rotation vector to rotation matrix
    std::cout.precision(3);
    std::cout << "rotation matrix = \n" << rotation_vector.matrix() << std::endl;
    rotation_matrix = rotation_vector.toRotationMatrix();

    // Convert coordinate with roation vector
    Eigen::Vector3d v(1, 0, 0);
    Eigen::Vector3d v_rotated = rotation_vector * v;
    std::cout << "(1, 0, 0) after rotation (by matrix) = " << v_rotated.transpose() << std::endl;

    // Euler angles, with the order ZYX
    Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0);
    std::cout << "yaw pitch roll = " << euler_angles.transpose() << std::endl;

    // Homogeneous transformation that uses Eigen::Isometry
    // It is called 3d but it actually is a 4 * 4 matrix
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    // Set rotate
    T.rotate(rotation_vector);
    // Set translate
    T.pretranslate(Eigen::Vector3d(1, 3, 4));
    std::cout << "Transform matrix = \n" << T.matrix() << std::endl;

    // Use T to preform coordinate transformation
    Eigen::Vector3d v_transformed = T * v;
    std::cout << "v transformed = " << v_transformed.transpose() << std::endl;

    // Quaternion
    Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector);
    // The order is (x, y, z, w)
    std::cout << "quaternion from rotation vector = " << q.coeffs().transpose() << std::endl;
    
    q = Eigen::Quaterniond(rotation_matrix);
    std::cout << "quaternion from rotation matrix = " << q.coeffs().transpose() << std::endl;

    // Transform a vector uses quaternion, noticed!! on math it is qvq^-1
    v_rotated = q * v;
    std::cout << "(1, 0, 0) after rotation = " << v_rotated.transpose() << std::endl;

    // Using math qvq^-1
    std::cout << "should be equal to  " << (q * Eigen::Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs().transpose() << std::endl;

    return 0;
}