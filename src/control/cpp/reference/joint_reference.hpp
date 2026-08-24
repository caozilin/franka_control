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

namespace franka_control::cpp {

class JointReferenceGenerator {
 public:
  JointReferenceGenerator(std::shared_ptr<const ReferenceGenerator> profile,
                          double duration,
                          const Vector7d& lower,
                          const Vector7d& upper)
      : profile_(std::move(profile)), duration_(duration), lower_(lower), upper_(upper) {
    if (profile_ == nullptr) throw std::invalid_argument("Joint reference profile cannot be null");
    if (std::string(profile_->name()) == "motion_limited") {
      throw std::invalid_argument(
          "motion_limited is not implemented for joint references; use min_jerk, linear, or cubic");
    }
  }

  void reset(const Vector7d& initial_q) {
    command_q_ = initial_q;
    segment_start_q_ = initial_q;
    segment_target_q_ = initial_q;
    segment_start_time_ = 0.0;
  }

  void acceptDelta(const std::array<double, 7>& delta, double elapsed) {
    bool has_action = false;
    for (double value : delta) has_action = has_action || std::abs(value) > 1e-12;
    if (!has_action) return;
    sample(elapsed);
    segment_start_q_ = command_q_;
    for (Eigen::Index i = 0; i < 7; ++i) {
      segment_target_q_[i] = std::clamp(segment_target_q_[i] + delta[static_cast<size_t>(i)], lower_[i], upper_[i]);
    }
    segment_start_time_ = elapsed;
  }

  void acceptTarget(const std::array<double, 7>& target, double elapsed) {
    sample(elapsed);
    segment_start_q_ = command_q_;
    for (Eigen::Index i = 0; i < 7; ++i) {
      segment_target_q_[i] = std::clamp(target[static_cast<size_t>(i)], lower_[i], upper_[i]);
    }
    segment_start_time_ = elapsed;
  }

  JointReferenceSample sample(double elapsed) {
    const ReferenceWeights weights = profile_->weights(
        (elapsed - segment_start_time_) / duration_, duration_);
    command_q_ = segment_start_q_ + weights.position * (segment_target_q_ - segment_start_q_);
    const Vector7d command_dq = weights.velocity * (segment_target_q_ - segment_start_q_);
    return {command_q_, command_dq};
  }

  const Vector7d& target() const { return segment_target_q_; }

 private:
  std::shared_ptr<const ReferenceGenerator> profile_;
  double duration_;
  Vector7d lower_;
  Vector7d upper_;
  Vector7d command_q_{Vector7d::Zero()};
  Vector7d segment_start_q_{Vector7d::Zero()};
  Vector7d segment_target_q_{Vector7d::Zero()};
  double segment_start_time_{0.0};
};

}  // namespace franka_control::cpp
