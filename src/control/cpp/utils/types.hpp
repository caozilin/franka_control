#pragma once

#include <Eigen/Dense>

namespace franka_control::cpp {

using Pose = Eigen::Matrix4d;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector7d = Eigen::Matrix<double, 7, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix67d = Eigen::Matrix<double, 6, 7>;
using Matrix76d = Eigen::Matrix<double, 7, 6>;
using Matrix7d = Eigen::Matrix<double, 7, 7>;

constexpr double kPolicyPeriod = 0.1;
constexpr double kGripperWidthMax = 0.08;
constexpr double kMaxTorqueRate = 1000.0;
constexpr double kPi = 3.14159265358979323846;
constexpr double kMaxRefLinearVelocity = 0.15;
constexpr double kMaxRefLinearAcceleration = 0.75;
constexpr double kMaxRefAngularVelocity = 3.0 * kPi / 8.0;
constexpr double kMaxRefAngularAcceleration = 15.0 * kPi / 8.0;
constexpr double kRefPositionEps = 0.0005;
constexpr double kRefLinearVelocityEps = 0.001;
constexpr double kRefRotationEps = 0.001;
constexpr double kRefAngularVelocityEps = 0.001;

}  // namespace franka_control::cpp
