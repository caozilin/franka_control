#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "utils/types.hpp"

namespace franka_control::cpp {

struct ReferenceWeights {
  double position;
  double velocity;
};

class ReferenceGenerator {
 public:
  virtual ~ReferenceGenerator() = default;
  virtual const char* name() const = 0;
  virtual ReferenceWeights weights(double raw_alpha) const = 0;
};

std::shared_ptr<ReferenceGenerator> makeReferenceGenerator(const std::string& name);

}  // namespace franka_control::cpp
