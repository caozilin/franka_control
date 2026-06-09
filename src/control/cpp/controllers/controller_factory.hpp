#pragma once

#include "controllers/controller_base.hpp"
#include "controllers/cubic_controller.hpp"
#include "controllers/linear_controller.hpp"
#include "controllers/min_jerk_controller.hpp"

namespace franka_control::cpp {

inline std::shared_ptr<ReferenceController> makeReferenceController(const std::string& name) {
  if (name == "min_jerk") return std::make_shared<MinJerkController>();
  if (name == "linear") return std::make_shared<LinearController>();
  if (name == "cubic") return std::make_shared<CubicController>();
  throw std::invalid_argument("controller_name must be one of: min_jerk, linear, cubic");
}

}  // namespace franka_control::cpp
