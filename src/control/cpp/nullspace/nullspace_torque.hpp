#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include "utils/types.hpp"

namespace franka_control::cpp {

enum class PseudoInverseMode {
  kPlain,
  kDamped,
};

enum class NullspaceProjectorMode {
  kKinematic,
  kDynamic,
};

struct NullspaceConfig {
  bool enabled{false};
  PseudoInverseMode pinv_mode{PseudoInverseMode::kPlain};
  NullspaceProjectorMode projector_mode{NullspaceProjectorMode::kKinematic};
  double damping_lambda{0.05};
  double stiffness{10.0};
  double damping{2.0};
  std::array<double, 7> q_target{};
};

inline PseudoInverseMode parsePseudoInverseMode(const std::string& mode) {
  if (mode == "plain") return PseudoInverseMode::kPlain;
  if (mode == "damped") return PseudoInverseMode::kDamped;
  throw std::invalid_argument("nullspace_pinv must be one of: plain, damped");
}

inline NullspaceProjectorMode parseNullspaceProjectorMode(const std::string& mode) {
  if (mode == "kinematic") return NullspaceProjectorMode::kKinematic;
  if (mode == "dynamic") return NullspaceProjectorMode::kDynamic;
  throw std::invalid_argument("nullspace_projector must be one of: kinematic, dynamic");
}

inline Matrix6d maybeDampMatrix6(const Matrix6d& value, PseudoInverseMode mode, double lambda) {
  if (mode == PseudoInverseMode::kPlain) return value;
  const double safe_lambda = std::max(0.0, lambda);
  return value + safe_lambda * safe_lambda * Matrix6d::Identity();
}

inline double pseudoInverseSingularValue(double singular_value, const NullspaceConfig& config) {
  static constexpr double kPlainTolerance = 1e-6;
  if (config.pinv_mode == PseudoInverseMode::kPlain) {
    return singular_value > kPlainTolerance ? 1.0 / singular_value : 0.0;
  }

  const double safe_lambda = std::max(0.0, config.damping_lambda);
  return singular_value / (singular_value * singular_value + safe_lambda * safe_lambda);
}

inline Matrix76d pseudoInverseJacobian(const Matrix67d& jacobian, const NullspaceConfig& config) {
  Eigen::JacobiSVD<Matrix67d> svd(jacobian, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Matrix76d sigma_pinv = Matrix76d::Zero();
  const auto singular_values = svd.singularValues();
  for (Eigen::Index i = 0; i < singular_values.size(); ++i) {
    sigma_pinv(i, i) = pseudoInverseSingularValue(singular_values(i), config);
  }
  return svd.matrixV() * sigma_pinv * svd.matrixU().transpose();
}

inline Matrix6d pseudoInverseMatrix6(const Matrix6d& matrix, const NullspaceConfig& config) {
  Eigen::JacobiSVD<Matrix6d> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Matrix6d sigma_pinv = Matrix6d::Zero();
  const auto singular_values = svd.singularValues();
  for (Eigen::Index i = 0; i < singular_values.size(); ++i) {
    sigma_pinv(i, i) = pseudoInverseSingularValue(singular_values(i), config);
  }
  return svd.matrixV() * sigma_pinv * svd.matrixU().transpose();
}

inline Matrix7d computeKinematicNullspaceProjector(const Matrix67d& jacobian, const NullspaceConfig& config) {
  const Matrix76d j_pinv = pseudoInverseJacobian(jacobian, config);
  return Matrix7d::Identity() - j_pinv * jacobian;
}

inline Matrix7d computeDynamicNullspaceProjector(const Matrix67d& jacobian,
                                                 const Matrix7d& mass,
                                                 const NullspaceConfig& config) {
  const Matrix7d mass_inv = mass.inverse();
  const Matrix6d lambda_inv = jacobian * mass_inv * jacobian.transpose();
  const Matrix6d inv = pseudoInverseMatrix6(lambda_inv, config);
  const Matrix76d j_bar = mass_inv * jacobian.transpose() * inv;
  return Matrix7d::Identity() - jacobian.transpose() * j_bar.transpose();
}

inline Vector7d computeNullspaceTorque(const Matrix67d& jacobian,
                                       const Matrix7d* mass,
                                       const Vector7d& q,
                                       const Vector7d& dq,
                                       const NullspaceConfig& config) {
  if (!config.enabled) return Vector7d::Zero();

  const Vector7d q_target = Eigen::Map<const Vector7d>(config.q_target.data());
  Vector7d tau_posture = Vector7d::Zero();
  for (Eigen::Index i = 0; i < tau_posture.size(); ++i) {
    if (std::isfinite(q_target(i))) {
      tau_posture(i) = config.stiffness * (q_target(i) - q(i)) - config.damping * dq(i);
    }
  }

  Vector7d tau_null = Vector7d::Zero();
  if (config.projector_mode == NullspaceProjectorMode::kKinematic) {
    tau_null = computeKinematicNullspaceProjector(jacobian, config) * tau_posture;
  } else {
  if (mass == nullptr) {
    throw std::invalid_argument("dynamic nullspace projector requires a mass matrix");
  }
    tau_null = computeDynamicNullspaceProjector(jacobian, *mass, config) * tau_posture;
  }

  return tau_null.allFinite() ? tau_null : Vector7d::Zero();
}

}  // namespace franka_control::cpp
