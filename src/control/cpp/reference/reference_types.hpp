#pragma once

#include "utils/types.hpp"

namespace franka_control::cpp {

struct CartesianReferenceSample {
  Pose target_pose{Pose::Identity()};
  Pose pose{Pose::Identity()};
  Vector6d velocity{Vector6d::Zero()};
};

struct JointReferenceSample {
  Vector7d q{Vector7d::Zero()};
  Vector7d dq{Vector7d::Zero()};
};

}  // namespace franka_control::cpp
