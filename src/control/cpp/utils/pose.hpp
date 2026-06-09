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
  const double trace = rotation.trace();
  const double cos_angle = std::clamp(0.5 * (trace - 1.0), -1.0, 1.0);
  const double angle = std::acos(cos_angle);
  if (angle < 1e-12) return Eigen::Vector3d::Zero();

  if (std::abs(angle - kPi) < 1e-6) {
    Eigen::Vector3d axis(
        std::sqrt(std::max(0.0, 0.5 * (rotation(0, 0) + 1.0))),
        std::sqrt(std::max(0.0, 0.5 * (rotation(1, 1) + 1.0))),
        std::sqrt(std::max(0.0, 0.5 * (rotation(2, 2) + 1.0))));
    if (axis.x() > 1e-8) {
      axis.y() = (rotation(0, 1) + rotation(1, 0)) / (4.0 * axis.x());
      axis.z() = (rotation(0, 2) + rotation(2, 0)) / (4.0 * axis.x());
    } else if (axis.y() > 1e-8) {
      axis.x() = (rotation(0, 1) + rotation(1, 0)) / (4.0 * axis.y());
      axis.z() = (rotation(1, 2) + rotation(2, 1)) / (4.0 * axis.y());
    } else if (axis.z() > 1e-8) {
      axis.x() = (rotation(0, 2) + rotation(2, 0)) / (4.0 * axis.z());
      axis.y() = (rotation(1, 2) + rotation(2, 1)) / (4.0 * axis.z());
    }
    const double norm = axis.norm();
    if (norm < 1e-12) return Eigen::Vector3d::Zero();
    return angle * axis / norm;
  }

  Eigen::Vector3d axis(rotation(2, 1) - rotation(1, 2), rotation(0, 2) - rotation(2, 0),
                       rotation(1, 0) - rotation(0, 1));
  axis /= 2.0 * std::sin(angle);
  return axis * angle;
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
  if (std::abs(angle - kPi) < 1e-4 && previous_norm >= 1e-12 && axis.dot(previous / previous_norm) < 0.0) {
    axis = -axis;
    rotvec = angle * axis;
  }
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

inline Vector6d poseError(const std::array<double, 16>& current_pose,
                          const Pose& desired_pose,
                          const Eigen::Vector3d& previous_rotvec) {
  const Pose current = poseFromArray(current_pose);
  Vector6d error = Vector6d::Zero();
  error.head<3>() = current.block<3, 1>(0, 3) - desired_pose.block<3, 1>(0, 3);
  error.tail<3>() =
      matrixToRotvecContinuous(current.block<3, 3>(0, 0) * desired_pose.block<3, 3>(0, 0).transpose(), previous_rotvec);
  return error;
}

}  // namespace franka_control::cpp
