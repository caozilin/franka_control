#pragma once

#include <chrono>
#include <utility>

#include "nullspace/nullspace_torque.hpp"
#include "reference/reference_types.hpp"
#include "utils/pose.hpp"
#include "utils/timing.hpp"

namespace franka_control::cpp {

struct CartesianTrackerTiming {
  double pose_error{0.0};
  double wrench_torque{0.0};
  double nullspace_torque{0.0};
};

struct CartesianTrackingOutput {
  Vector6d pose_error{Vector6d::Zero()};
  Vector6d task_wrench{Vector6d::Zero()};
  Vector7d task_torque{Vector7d::Zero()};
  Vector7d nullspace_torque{Vector7d::Zero()};
  Vector7d desired_torque{Vector7d::Zero()};
};

class CartesianImpedanceTracker {
 public:
  CartesianImpedanceTracker(Matrix6d stiffness, Matrix6d damping, NullspaceConfig nullspace)
      : stiffness_(std::move(stiffness)), damping_(std::move(damping)), nullspace_(std::move(nullspace)) {}

  CartesianTrackingOutput compute(const Pose& actual_pose,
                                  const Vector7d& q,
                                  const Vector7d& dq,
                                  const Vector7d& coriolis,
                                  const Matrix67d& jacobian,
                                  const Matrix7d* mass,
                                  const CartesianReferenceSample& reference,
                                  CartesianTrackerTiming* timing = nullptr) {
    CartesianTrackingOutput output;
    auto started = Clock::now();
    output.pose_error = poseError(actual_pose, reference.pose, last_error_rotvec_);
    last_error_rotvec_ = output.pose_error.tail<3>();
    if (timing != nullptr) timing->pose_error += secondsSince(started);

    started = Clock::now();
    output.task_wrench = maskTaskVector(
        -stiffness_ * output.pose_error + damping_ * (reference.velocity - jacobian * dq), nullspace_);
    output.task_torque = jacobian.transpose() * output.task_wrench;
    if (timing != nullptr) timing->wrench_torque += secondsSince(started);

    started = Clock::now();
    output.nullspace_torque = computeNullspaceTorque(jacobian, mass, q, dq, nullspace_);
    output.desired_torque = output.task_torque + output.nullspace_torque + coriolis;
    if (timing != nullptr) timing->nullspace_torque += secondsSince(started);
    return output;
  }

 private:
  Matrix6d stiffness_;
  Matrix6d damping_;
  NullspaceConfig nullspace_;
  Eigen::Vector3d last_error_rotvec_{Eigen::Vector3d::Zero()};
};

}  // namespace franka_control::cpp
