#pragma once

#include <algorithm>

#include "reference/reference_base.hpp"

namespace franka_control::cpp {

class CubicReferenceGenerator final : public ReferenceGenerator {
 public:
  const char* name() const override { return "cubic"; }

  ReferenceWeights weights(double raw_alpha, double duration) const override {
    const double alpha = std::clamp(raw_alpha, 0.0, 1.0);
    return {3.0 * alpha * alpha - 2.0 * alpha * alpha * alpha,
            (6.0 * alpha - 6.0 * alpha * alpha) / duration};
  }
};

}  // namespace franka_control::cpp
