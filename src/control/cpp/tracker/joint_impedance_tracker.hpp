#pragma once

#include <cmath>

#include "reference/reference_types.hpp"
#include "utils/types.hpp"

namespace franka_control::cpp {

class JointImpedanceTracker {
 public:
  JointImpedanceTracker() {
    stiffness_ << 80.0, 80.0, 80.0, 60.0, 25.0, 15.0, 10.0;
    for (Eigen::Index i = 0; i < 7; ++i) damping_[i] = 2.0 * std::sqrt(stiffness_[i]);
  }

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
