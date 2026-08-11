#pragma once

#include <algorithm>

#include "reference/reference_base.hpp"

namespace franka_control::cpp {

class LinearReferenceGenerator final : public ReferenceGenerator {
 public:
  const char* name() const override { return "linear"; }

  ReferenceWeights weights(double raw_alpha, double duration) const override {
    const double alpha = std::clamp(raw_alpha, 0.0, 1.0);
    const double velocity = (raw_alpha >= 0.0 && raw_alpha <= 1.0) ? 1.0 / duration : 0.0;
    return {alpha, velocity};
  }
};

}  // namespace franka_control::cpp
