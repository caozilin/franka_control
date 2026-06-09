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

class ReferenceController {
 public:
  virtual ~ReferenceController() = default;
  virtual const char* name() const = 0;
  virtual ReferenceWeights weights(double raw_alpha) const = 0;
};

std::shared_ptr<ReferenceController> makeReferenceController(const std::string& name);

}  // namespace franka_control::cpp
