#pragma once

#include "reference/reference_base.hpp"

namespace franka_control::cpp {

class MotionLimitedReferenceGenerator final : public ReferenceGenerator {
 public:
  const char* name() const override { return "motion_limited"; }

  ReferenceWeights weights(double raw_alpha) const override {
    (void)raw_alpha;
    return {0.0, 0.0};
  }
};

}  // namespace franka_control::cpp
