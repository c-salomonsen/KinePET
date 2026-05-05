#pragma once

#include "solver.hpp"
#include "tcm_model.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tcm {

struct RescueRefitOptions {
  bool enable = false;
  int radius = 1;
  int min_reliable_neighbors = 3;
  int iterations = 12;
  double high_rmse_quantile = 0.75;
};

struct RescueRefitResult {
  std::vector<int32_t> target_mask;
  std::vector<int32_t> applied_mask;
  std::vector<int32_t> unresolved_mask;
  std::size_t targeted_voxels = 0;
  std::size_t applied_voxels = 0;
};

RescueRefitResult apply_rescue_refit(
    const SolverOptions& solver_options,
    const RescueRefitOptions& rescue_options,
    const std::vector<int32_t>& spatial_mask,
    std::size_t depth,
    std::size_t height,
    std::size_t width,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif,
    const std::vector<double>& weights,
    const std::vector<double>& fit_image,
    std::vector<double>& k1_map,
    std::vector<double>& k2_map,
    std::vector<double>& k3_map,
    std::vector<double>* k4_map,
    std::vector<double>& vb_map,
    std::vector<double>* delay_map,
    std::vector<double>* dispersion_map,
    std::vector<double>& ki_map,
    std::vector<double>& rmse_map,
    std::vector<int32_t>& iter_map,
    std::vector<int32_t>& status_map);

}  // namespace tcm
