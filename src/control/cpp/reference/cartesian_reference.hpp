#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "reference/reference_base.hpp"
#include "reference/reference_types.hpp"
#include "utils/control.hpp"
#include "utils/pose.hpp"

namespace franka_control::cpp {

class CartesianReferenceGenerator {
 public:
  CartesianReferenceGenerator(std::shared_ptr<const ReferenceGenerator> profile,
                              double max_translation_step,
                              double max_rotation_step,
                              double max_translation_velocity,
                              double max_rotation_velocity,
                              double max_translation_acceleration,
                              double max_rotation_acceleration,
                              double segment_duration,
                              double position_epsilon,
                              double linear_velocity_epsilon,
                              double rotation_epsilon,
                              double angular_velocity_epsilon)
      : profile_(std::move(profile)),
        max_translation_step_(max_translation_step),
        max_rotation_step_(max_rotation_step),
        max_translation_velocity_(max_translation_velocity),
        max_rotation_velocity_(max_rotation_velocity),
        max_translation_acceleration_(max_translation_acceleration),
        max_rotation_acceleration_(max_rotation_acceleration),
        segment_duration_(segment_duration),
        position_epsilon_(position_epsilon),
        linear_velocity_epsilon_(linear_velocity_epsilon),
        rotation_epsilon_(rotation_epsilon),
        angular_velocity_epsilon_(angular_velocity_epsilon) {
    if (profile_ == nullptr) throw std::invalid_argument("Cartesian reference profile cannot be null");
    motion_limited_ = std::string(profile_->name()) == "motion_limited";
  }

  void reset(const Pose& initial_pose) {
    command_pose_ = initial_pose;
    segment_start_pose_ = initial_pose;
    segment_target_pose_ = initial_pose;
    segment_delta_translation_.setZero();
    segment_delta_rotvec_.setZero();
    last_segment_rotvec_.setZero();
    linear_velocity_.setZero();
    linear_acceleration_.setZero();
    angular_velocity_.setZero();
    angular_acceleration_.setZero();
    last_rotation_error_.setZero();
    segment_start_time_ = 0.0;
  }

  void acceptAction(const std::array<double, 7>& action, double elapsed) {
    segment_start_pose_ = command_pose_;
    Pose candidate = segment_target_pose_;
    candidate.block<3, 1>(0, 3) += transformTranslation(action, max_translation_step_);
    candidate.block<3, 3>(0, 0) =
        rotvecToMatrix(transformRotation(action, max_rotation_step_)) * candidate.block<3, 3>(0, 0);
    segment_target_pose_ = candidate;
    segment_delta_translation_ =
        segment_target_pose_.block<3, 1>(0, 3) - segment_start_pose_.block<3, 1>(0, 3);
    segment_delta_rotvec_ = matrixToRotvecContinuous(
        segment_target_pose_.block<3, 3>(0, 0) * segment_start_pose_.block<3, 3>(0, 0).transpose(),
        last_segment_rotvec_);
    last_segment_rotvec_ = segment_delta_rotvec_;
    segment_start_time_ = elapsed;
  }

  CartesianReferenceSample sample(double elapsed, double dt) {
    Vector6d desired_velocity = Vector6d::Zero();
    if (isMotionLimited()) {
      updateMotionLimited(dt, &desired_velocity);
    } else {
      const auto weights = profile_->weights(
          (elapsed - segment_start_time_) / segment_duration_, segment_duration_);
      command_pose_ = segment_start_pose_;
      command_pose_.block<3, 1>(0, 3) += weights.position * segment_delta_translation_;
      command_pose_.block<3, 3>(0, 0) =
          rotvecToMatrix(weights.position * segment_delta_rotvec_) * segment_start_pose_.block<3, 3>(0, 0);
      desired_velocity.head<3>() = weights.velocity * segment_delta_translation_;
      desired_velocity.tail<3>() = weights.velocity * segment_delta_rotvec_;
    }
    return {segment_target_pose_, command_pose_, desired_velocity};
  }

 private:
  static Eigen::Vector3d clampNorm(const Eigen::Vector3d& value, double limit) {
    const double norm = value.norm();
    if (norm <= limit || norm < 1e-12) return value;
    return value * (limit / norm);
  }

  bool isMotionLimited() const { return motion_limited_; }

  void updateMotionLimited(double dt, Vector6d* desired_velocity) {
    desired_velocity->setZero();
    const Eigen::Vector3d position_goal = segment_target_pose_.block<3, 1>(0, 3);
    const Eigen::Vector3d position_reference = command_pose_.block<3, 1>(0, 3);
    const Eigen::Vector3d position_error = position_goal - position_reference;
    const double position_distance = position_error.norm();

    if (position_distance < position_epsilon_ && linear_velocity_.norm() < linear_velocity_epsilon_) {
      command_pose_.block<3, 1>(0, 3) = position_goal;
      linear_velocity_.setZero();
      linear_acceleration_.setZero();
    } else {
      Eigen::Vector3d desired = Eigen::Vector3d::Zero();
      if (position_distance >= 1e-12) {
        const Eigen::Vector3d direction = position_error / position_distance;
        const double stopping_limit =
            std::sqrt(std::max(0.0, 2.0 * max_translation_acceleration_ * position_distance));
        desired = direction * std::min({max_translation_velocity_, stopping_limit, position_distance / dt});
      }
      linear_acceleration_ = clampNorm((desired - linear_velocity_) / dt, max_translation_acceleration_);
      linear_velocity_ = clampNorm(linear_velocity_ + linear_acceleration_ * dt, max_translation_velocity_);
      const Eigen::Vector3d next = position_reference + linear_velocity_ * dt;
      if (position_distance >= 1e-12 && position_error.dot(position_goal - next) <= 0.0) {
        command_pose_.block<3, 1>(0, 3) = position_goal;
        linear_velocity_.setZero();
        linear_acceleration_.setZero();
      } else {
        command_pose_.block<3, 1>(0, 3) = next;
      }
    }

    const Eigen::Vector3d rotation_error = matrixToRotvecContinuous(
        segment_target_pose_.block<3, 3>(0, 0) * command_pose_.block<3, 3>(0, 0).transpose(),
        last_rotation_error_);
    last_rotation_error_ = rotation_error;
    const double rotation_distance = rotation_error.norm();
    if (rotation_distance < rotation_epsilon_ && angular_velocity_.norm() < angular_velocity_epsilon_) {
      command_pose_.block<3, 3>(0, 0) = segment_target_pose_.block<3, 3>(0, 0);
      angular_velocity_.setZero();
      angular_acceleration_.setZero();
      last_rotation_error_.setZero();
    } else {
      Eigen::Vector3d desired = Eigen::Vector3d::Zero();
      if (rotation_distance >= 1e-12) {
        const Eigen::Vector3d direction = rotation_error / rotation_distance;
        const double stopping_limit =
            std::sqrt(std::max(0.0, 2.0 * max_rotation_acceleration_ * rotation_distance));
        desired = direction * std::min({max_rotation_velocity_, stopping_limit, rotation_distance / dt});
      }
      angular_acceleration_ = clampNorm((desired - angular_velocity_) / dt, max_rotation_acceleration_);
      angular_velocity_ = clampNorm(angular_velocity_ + angular_acceleration_ * dt, max_rotation_velocity_);
      const Eigen::Vector3d step = angular_velocity_ * dt;
      if (rotation_distance >= 1e-12 && rotation_error.dot(step) >= rotation_distance * rotation_distance) {
        command_pose_.block<3, 3>(0, 0) = segment_target_pose_.block<3, 3>(0, 0);
        angular_velocity_.setZero();
        angular_acceleration_.setZero();
        last_rotation_error_.setZero();
      } else {
        command_pose_.block<3, 3>(0, 0) = rotvecToMatrix(step) * command_pose_.block<3, 3>(0, 0);
      }
    }
    desired_velocity->head<3>() = linear_velocity_;
    desired_velocity->tail<3>() = angular_velocity_;
  }

  std::shared_ptr<const ReferenceGenerator> profile_;
  bool motion_limited_{false};
  double max_translation_step_;
  double max_rotation_step_;
  double max_translation_velocity_;
  double max_rotation_velocity_;
  double max_translation_acceleration_;
  double max_rotation_acceleration_;
  double segment_duration_;
  double position_epsilon_;
  double linear_velocity_epsilon_;
  double rotation_epsilon_;
  double angular_velocity_epsilon_;
  Pose command_pose_{Pose::Identity()};
  Pose segment_start_pose_{Pose::Identity()};
  Pose segment_target_pose_{Pose::Identity()};
  Eigen::Vector3d segment_delta_translation_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d segment_delta_rotvec_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d last_segment_rotvec_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d linear_velocity_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d linear_acceleration_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d angular_velocity_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d angular_acceleration_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d last_rotation_error_{Eigen::Vector3d::Zero()};
  double segment_start_time_{0.0};
};

}  // namespace franka_control::cpp
