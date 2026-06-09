#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "utils/types.hpp"

namespace franka_control::cpp {

inline Eigen::Vector3d transformTranslation(const std::array<double, 7>& action, double max_translation_step) {
  return Eigen::Vector3d{
      std::clamp(action[0], -1.0, 1.0) * max_translation_step,
      std::clamp(action[1], -1.0, 1.0) * max_translation_step,
      std::clamp(action[2], -1.0, 1.0) * max_translation_step,
  };
}

inline Eigen::Vector3d transformRotation(const std::array<double, 7>& action, double max_rotation_step) {
  return Eigen::Vector3d{
      std::clamp(action[3], -6.0, 6.0) * max_rotation_step / 6.0,
      std::clamp(action[4], -6.0, 6.0) * max_rotation_step / 6.0,
      std::clamp(action[5], -6.0, 6.0) * max_rotation_step / 6.0,
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

inline Vector7d limitTorqueRate(const Vector7d& tau_d, const std::array<double, 7>& tau_j_d, double dt) {
  const double max_delta = kMaxTorqueRate * std::max(dt, 0.001);
  const Vector7d previous = vector7FromArray(tau_j_d);
  Vector7d out = previous;
  for (int i = 0; i < 7; ++i) {
    out(i) += std::clamp(tau_d(i) - previous(i), -max_delta, max_delta);
  }
  return out;
}

inline Matrix6d defaultStiffness() {
  Matrix6d stiffness = Matrix6d::Zero();
  stiffness.diagonal() << 600.0, 600.0, 600.0, 50.0, 50.0, 50.0;
  return stiffness;
}

inline Matrix6d defaultDamping() {
  Matrix6d damping = Matrix6d::Zero();
  const Matrix6d stiffness = defaultStiffness();
  for (int i = 0; i < 6; ++i) damping(i, i) = 2.0 * std::sqrt(stiffness(i, i));
  return damping;
}

}  // namespace franka_control::cpp
