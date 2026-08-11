#pragma once

#include <utility>

#include "reference/reference_types.hpp"
#include "utils/types.hpp"

namespace franka_control::cpp {

class JointImpedanceTracker {
 public:
  JointImpedanceTracker(Vector7d stiffness, Vector7d damping)
      : stiffness_(std::move(stiffness)), damping_(std::move(damping)) {}

  Vector7d compute(const Vector7d& q,
                   const Vector7d& dq,
                   const Vector7d& coriolis,
                   const JointReferenceSample& reference) const {
    return stiffness_.cwiseProduct(reference.q - q) + damping_.cwiseProduct(reference.dq - dq) + coriolis;
  }

 private:
  Vector7d stiffness_{Vector7d::Zero()};
  Vector7d damping_{Vector7d::Zero()};
};

}  // namespace franka_control::cpp
