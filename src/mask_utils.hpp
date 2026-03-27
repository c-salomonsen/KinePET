#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tcm {

struct MaskOptions {
  double threshold = 0.05;
  bool keep_largest_component = true;
  bool fill_holes = true;
};

struct MaskResult {
  std::vector<int32_t> raw_mask;
  std::vector<int32_t> cleaned_mask;
  std::vector<int32_t> removed_mask;
  std::vector<int32_t> filled_mask;
};

MaskResult build_and_clean_mask(
    const std::vector<double>& image,
    std::size_t t_count,
    std::size_t depth,
    std::size_t height,
    std::size_t width,
    const MaskOptions& options);

}  // namespace tcm
