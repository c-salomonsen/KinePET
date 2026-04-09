#pragma once

#include "solver.hpp"
#include "tcm_model.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tcm {

struct PostFilterOptions {
  bool enable_status_median = true;
  int radius = 1;
  int min_reliable_neighbors = 3;
};

struct PostFilterResult {
  std::vector<int32_t> target_mask;
  std::vector<int32_t> applied_mask;
  std::vector<int32_t> unresolved_mask;
  std::size_t targeted_voxels = 0;
  std::size_t applied_voxels = 0;
};

PostFilterResult apply_selective_status_median_filter(
    ModelKind model,
    const PostFilterOptions& options,
    const std::vector<int32_t>& spatial_mask,
    const std::vector<int32_t>& status_map,
    std::size_t depth,
    std::size_t height,
    std::size_t width,
    std::vector<double>& k1_map,
    std::vector<double>& k2_map,
    std::vector<double>& k3_map,
    std::vector<double>* k4_map,
    std::vector<double>& vb_map,
    std::vector<double>* delay_map,
    std::vector<double>* dispersion_map,
    std::vector<double>& ki_map);

}  // namespace tcm
