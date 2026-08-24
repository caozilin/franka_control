#pragma once

#include <array>

#include "utils/control.hpp"

namespace franka_control::cpp {

class TorqueRateLimiter {
 public:
  explicit TorqueRateLimiter(double max_torque_rate) : max_torque_rate_(max_torque_rate) {}

  Vector7d apply(const Vector7d& desired,
                 const std::array<double, 7>& previous_desired) const {
    return limitTorqueRate(desired, previous_desired, max_torque_rate_);
  }

 private:
  double max_torque_rate_;
};

}  // namespace franka_control::cpp
