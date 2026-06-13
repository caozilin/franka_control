#pragma once

#include <algorithm>

#include "reference/reference_base.hpp"

namespace franka_control::cpp {

class MinJerkReferenceGenerator final : public ReferenceGenerator {
 public:
  const char* name() const override { return "min_jerk"; }

  ReferenceWeights weights(double raw_alpha) const override {
    const double alpha = std::clamp(raw_alpha, 0.0, 1.0);
    const double alpha2 = alpha * alpha;
    const double alpha3 = alpha2 * alpha;
    const double alpha4 = alpha3 * alpha;
    const double alpha5 = alpha4 * alpha;
    return {10.0 * alpha3 - 15.0 * alpha4 + 6.0 * alpha5,
            (30.0 * alpha2 - 60.0 * alpha3 + 30.0 * alpha4) / kPolicyPeriod};
  }
};

}  // namespace franka_control::cpp
