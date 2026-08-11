#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "reference/reference_types.hpp"

namespace franka_control::cpp {

class JointMinJerkReferenceGenerator {
 public:
  JointMinJerkReferenceGenerator(double duration, const Vector7d& lower, const Vector7d& upper)
      : duration_(duration), lower_(lower), upper_(upper) {}

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
    segment_start_q_ = command_q_;
    for (Eigen::Index i = 0; i < 7; ++i) {
      segment_target_q_[i] = std::clamp(segment_target_q_[i] + delta[static_cast<size_t>(i)], lower_[i], upper_[i]);
    }
    segment_start_time_ = elapsed;
  }

  void acceptTarget(const std::array<double, 7>& target, double elapsed) {
    segment_start_q_ = command_q_;
    for (Eigen::Index i = 0; i < 7; ++i) {
      segment_target_q_[i] = std::clamp(target[static_cast<size_t>(i)], lower_[i], upper_[i]);
    }
    segment_start_time_ = elapsed;
  }

  JointReferenceSample sample(double elapsed) {
    const double alpha = std::clamp((elapsed - segment_start_time_) / duration_, 0.0, 1.0);
    const double alpha2 = alpha * alpha;
    const double alpha3 = alpha2 * alpha;
    const double alpha4 = alpha3 * alpha;
    const double alpha5 = alpha4 * alpha;
    const double weight = 10.0 * alpha3 - 15.0 * alpha4 + 6.0 * alpha5;
    const double derivative =
        alpha >= 1.0 ? 0.0 : (30.0 * alpha2 - 60.0 * alpha3 + 30.0 * alpha4) / duration_;
    command_q_ = segment_start_q_ + weight * (segment_target_q_ - segment_start_q_);
    const Vector7d command_dq = derivative * (segment_target_q_ - segment_start_q_);
    return {command_q_, command_dq};
  }

 private:
  double duration_;
  Vector7d lower_;
  Vector7d upper_;
  Vector7d command_q_{Vector7d::Zero()};
  Vector7d segment_start_q_{Vector7d::Zero()};
  Vector7d segment_target_q_{Vector7d::Zero()};
  double segment_start_time_{0.0};
};

}  // namespace franka_control::cpp
