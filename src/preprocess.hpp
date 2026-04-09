#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace tcm {

enum class TemporalFilterKind {
  None,
  Median,
  Gaussian,
};

struct PreprocessingOptions {
  bool spatial_gaussian_enable = false;
  int spatial_gaussian_radius = 1;
  double spatial_gaussian_sigma = 0.75;
  TemporalFilterKind temporal_filter = TemporalFilterKind::None;
  int temporal_filter_radius = 1;
  double temporal_gaussian_sigma = 1.0;
};

std::string temporal_filter_name(TemporalFilterKind kind);

std::vector<double> apply_prefit_preprocessing(
    const std::vector<double>& image,
    std::size_t t_count,
    std::size_t depth,
    std::size_t height,
    std::size_t width,
    const PreprocessingOptions& options);

}  // namespace tcm
