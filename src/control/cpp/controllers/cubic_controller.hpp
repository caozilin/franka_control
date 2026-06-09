#pragma once

#include <algorithm>

#include "controllers/controller_base.hpp"

namespace franka_control::cpp {

class CubicController final : public ReferenceController {
 public:
  const char* name() const override { return "cubic"; }

  ReferenceWeights weights(double raw_alpha) const override {
    const double alpha = std::clamp(raw_alpha, 0.0, 1.0);
    return {3.0 * alpha * alpha - 2.0 * alpha * alpha * alpha,
            (6.0 * alpha - 6.0 * alpha * alpha) / kPolicyPeriod};
  }
};

}  // namespace franka_control::cpp
