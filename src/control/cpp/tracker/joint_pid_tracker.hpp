#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "reference/reference_types.hpp"
#include "tracker/joint_impedance_tracker.hpp"
#include "utils/types.hpp"

namespace franka_control::cpp {

struct JointPidSettings {
  double proportional_gain{0.18};
  double integral_gain_s{0.30};
  double velocity_gain_s{0.04};
  double maximum_correction_rad{0.05235987755982989};
  double integration_error_limit_rad{0.06981317007977318};
  double integral_time_constant_s{1.0};
  double stationary_integral_time_constant_s{0.25};
  double stationary_velocity_threshold_rad_s{0.02};
};

class JointPidTracker {
 public:
  JointPidTracker(Vector7d joint_lower,
                  Vector7d joint_upper,
                  Vector7d stiffness,
                  Vector7d damping,
                  JointPidSettings settings)
      : joint_lower_(std::move(joint_lower)),
        joint_upper_(std::move(joint_upper)),
        impedance_(std::move(stiffness), std::move(damping)),
        settings_(settings) {
    validateSettings();
  }

  void reset() { integral_error_.setZero(); }

  Vector7d compute(const Vector7d& q,
                   const Vector7d& dq,
                   const Vector7d& coriolis,
                   const JointReferenceSample& reference,
                   double dt) {
    if (!std::isfinite(dt) || dt <= 0.0) {
      throw std::invalid_argument("joint PID dt must be finite and positive");
    }
    const Vector7d error = reference.q - q;
    const Vector7d velocity_error = reference.dq - dq;
    const bool stationary =
        reference.dq.cwiseAbs().maxCoeff() <= settings_.stationary_velocity_threshold_rad_s;
    const double time_constant = stationary
                                     ? settings_.stationary_integral_time_constant_s
                                     : settings_.integral_time_constant_s;
    const double decay = std::exp(-dt / time_constant);

    Vector7d proposed_integral = decay * integral_error_;
    for (Eigen::Index i = 0; i < proposed_integral.size(); ++i) {
      proposed_integral(i) +=
          std::clamp(error(i), -settings_.integration_error_limit_rad,
                     settings_.integration_error_limit_rad) *
          dt;
    }
    if (settings_.integral_gain_s > 0.0) {
      const double integral_limit =
          settings_.maximum_correction_rad / settings_.integral_gain_s;
      proposed_integral = proposed_integral.cwiseMax(-integral_limit).cwiseMin(integral_limit);
    }

    const Vector7d raw_correction =
        settings_.proportional_gain * error +
        settings_.integral_gain_s * proposed_integral +
        settings_.velocity_gain_s * velocity_error;
    const Vector7d correction =
        raw_correction.cwiseMax(-settings_.maximum_correction_rad)
            .cwiseMin(settings_.maximum_correction_rad);
    const Vector7d unclipped_command = reference.q + correction;
    const Vector7d command = unclipped_command.cwiseMax(joint_lower_).cwiseMin(joint_upper_);

    for (Eigen::Index i = 0; i < integral_error_.size(); ++i) {
      const bool saturated =
          std::abs(raw_correction(i) - correction(i)) > 1e-12 ||
          std::abs(unclipped_command(i) - command(i)) > 1e-12;
      const bool same_direction =
          (error(i) > 0.0 && raw_correction(i) > 0.0) ||
          (error(i) < 0.0 && raw_correction(i) < 0.0);
      if (!(saturated && same_direction)) integral_error_(i) = proposed_integral(i);
    }

    JointReferenceSample corrected_reference{command, reference.dq};
    return impedance_.compute(q, dq, coriolis, corrected_reference);
  }

 private:
  void validateSettings() const {
    const double values[] = {
        settings_.proportional_gain,
        settings_.integral_gain_s,
        settings_.velocity_gain_s,
        settings_.maximum_correction_rad,
        settings_.integration_error_limit_rad,
        settings_.integral_time_constant_s,
        settings_.stationary_integral_time_constant_s,
        settings_.stationary_velocity_threshold_rad_s,
    };
    for (double value : values) {
      if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument("joint PID settings must be finite and non-negative");
      }
    }
    if (settings_.maximum_correction_rad <= 0.0 ||
        settings_.integral_time_constant_s <= 0.0 ||
        settings_.stationary_integral_time_constant_s <= 0.0) {
      throw std::invalid_argument("joint PID correction and time constants must be positive");
    }
  }

  Vector7d joint_lower_;
  Vector7d joint_upper_;
  JointImpedanceTracker impedance_;
  JointPidSettings settings_;
  Vector7d integral_error_{Vector7d::Zero()};
};

}  // namespace franka_control::cpp
