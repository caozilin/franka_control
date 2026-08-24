#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include <franka/rate_limiting.h>

#include "utils/types.hpp"

namespace franka_control::cpp {

constexpr double kTorqueControlPeriodS = 1e-3;

inline Eigen::Vector3d transformTranslation(const std::array<double, 7>& action, double max_translation_step) {
  return Eigen::Vector3d{
      std::clamp(action[0], -max_translation_step, max_translation_step),
      std::clamp(action[1], -max_translation_step, max_translation_step),
      std::clamp(action[2], -max_translation_step, max_translation_step),
  };
}

inline Eigen::Vector3d transformRotation(const std::array<double, 7>& action, double max_rotation_step) {
  return Eigen::Vector3d{
      std::clamp(action[3], -max_rotation_step, max_rotation_step),
      std::clamp(action[4], -max_rotation_step, max_rotation_step),
      std::clamp(action[5], -max_rotation_step, max_rotation_step),
  };
}

inline Vector7d vector7FromArray(const std::array<double, 7>& input) {
  return Eigen::Map<const Vector7d>(input.data());
}

inline std::array<double, 7> arrayFromVector7(const Vector7d& input) {
  std::array<double, 7> out{};
  Eigen::Map<Vector7d>(out.data()) = input;
  return out;
}

inline Vector7d limitTorqueRate(const Vector7d& tau_d,
                                const std::array<double, 7>& tau_j_d,
                                double max_torque_rate) {
  const Vector7d previous = vector7FromArray(tau_j_d);
  Vector7d out = previous;
  for (int i = 0; i < 7; ++i) {
    const double max_delta =
        std::min(max_torque_rate, franka::kMaxTorqueRate[static_cast<size_t>(i)]) *
        kTorqueControlPeriodS;
    out(i) += std::clamp(tau_d(i) - previous(i), -max_delta, max_delta);
  }
  return out;
}

}  // namespace franka_control::cpp
