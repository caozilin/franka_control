#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "utils/types.hpp"

namespace franka_control::cpp {

inline Pose poseFromArray(const std::array<double, 16>& pose) {
  return Eigen::Map<const Pose>(pose.data());
}

inline Eigen::Matrix3d rotvecToMatrix(const Eigen::Vector3d& rotvec) {
  const double angle = rotvec.norm();
  if (angle < 1e-12) return Eigen::Matrix3d::Identity();
  return Eigen::AngleAxisd(angle, rotvec / angle).toRotationMatrix();
}

inline Eigen::Vector3d matrixToRotvec(const Eigen::Matrix3d& rotation) {
  Eigen::Quaterniond q(rotation);
  q.normalize();
  if (q.w() < 0.0) q.coeffs() *= -1.0;

  const double sin_half = q.vec().norm();
  if (sin_half < 1e-12) return Eigen::Vector3d::Zero();

  const double angle = 2.0 * std::atan2(sin_half, q.w());
  return angle * (q.vec() / sin_half);
}

inline Eigen::Vector3d matrixToRotvecContinuous(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& previous) {
  Eigen::Vector3d rotvec = matrixToRotvec(rotation);
  const double previous_norm = previous.norm();
  const double angle = rotvec.norm();
  if (previous_norm < 1e-12 && angle < 1e-12) return Eigen::Vector3d::Zero();

  if (angle < 1e-12) {
    const Eigen::Vector3d axis = previous_norm >= 1e-12 ? previous / previous_norm : Eigen::Vector3d{1.0, 0.0, 0.0};
    const int center = static_cast<int>(std::round(previous_norm / (2.0 * kPi)));
    Eigen::Vector3d best = Eigen::Vector3d::Zero();
    double best_distance = (best - previous).norm();
    for (int k = center - 3; k <= center + 3; ++k) {
      Eigen::Vector3d candidate = 2.0 * kPi * static_cast<double>(k) * axis;
      const double distance = (candidate - previous).norm();
      if (distance < best_distance) {
        best = candidate;
        best_distance = distance;
      }
    }
    return best;
  }

  Eigen::Vector3d axis = rotvec / angle;
  const double projected_previous = previous.dot(axis);
  const int center = static_cast<int>(std::round((projected_previous - angle) / (2.0 * kPi)));
  Eigen::Vector3d best = rotvec;
  double best_distance = (best - previous).norm();
  for (int k = center - 4; k <= center + 4; ++k) {
    Eigen::Vector3d candidate = rotvec + 2.0 * kPi * static_cast<double>(k) * axis;
    const double distance = (candidate - previous).norm();
    if (distance < best_distance) {
      best = candidate;
      best_distance = distance;
    }
  }
  return best;
}

inline Vector6d poseError(const Pose& current,
                         const Pose& desired_pose,
                         const Eigen::Vector3d& previous_rotvec) {
  (void)previous_rotvec;
  Vector6d error = Vector6d::Zero();
  error.head<3>() = current.block<3, 1>(0, 3) - desired_pose.block<3, 1>(0, 3);
  error.tail<3>() = matrixToRotvec(current.block<3, 3>(0, 0) * desired_pose.block<3, 3>(0, 0).transpose());
  return error;
}

inline Vector6d poseError(const std::array<double, 16>& current_pose,
                          const Pose& desired_pose,
                          const Eigen::Vector3d& previous_rotvec) {
  return poseError(poseFromArray(current_pose), desired_pose, previous_rotvec);
}

}  // namespace franka_control::cpp
