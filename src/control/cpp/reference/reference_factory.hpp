#pragma once

#include "reference/reference_base.hpp"
#include "reference/cubic_reference.hpp"
#include "reference/linear_reference.hpp"
#include "reference/min_jerk_reference.hpp"
#include "reference/motion_limited_reference.hpp"

namespace franka_control::cpp {

inline std::shared_ptr<ReferenceGenerator> makeReferenceGenerator(const std::string& name) {
  if (name == "min_jerk") return std::make_shared<MinJerkReferenceGenerator>();
  if (name == "linear") return std::make_shared<LinearReferenceGenerator>();
  if (name == "cubic") return std::make_shared<CubicReferenceGenerator>();
  if (name == "motion_limited") return std::make_shared<MotionLimitedReferenceGenerator>();
  throw std::invalid_argument("reference_name must be one of: min_jerk, linear, cubic, motion_limited");
}

}  // namespace franka_control::cpp
