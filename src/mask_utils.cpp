#include "mask_utils.hpp"

#include <algorithm>
#include <array>
#include <deque>
#include <stdexcept>

namespace tcm {
namespace {

std::size_t flatten_index(
    std::size_t z,
    std::size_t y,
    std::size_t x,
    std::size_t height,
    std::size_t width) {
  return (z * height + y) * width + x;
}

void push_neighbors_6(
    std::size_t index,
    std::size_t depth,
    std::size_t height,
    std::size_t width,
    std::array<std::size_t, 6>& neighbors,
    int& count) {
  count = 0;
  const std::size_t plane = height * width;
  const std::size_t z = index / plane;
  const std::size_t rem = index % plane;
  const std::size_t y = rem / width;
  const std::size_t x = rem % width;

  if (z > 0) {
    neighbors[static_cast<std::size_t>(count++)] = index - plane;
  }
  if (z + 1 < depth) {
    neighbors[static_cast<std::size_t>(count++)] = index + plane;
  }
  if (y > 0) {
    neighbors[static_cast<std::size_t>(count++)] = index - width;
  }
  if (y + 1 < height) {
    neighbors[static_cast<std::size_t>(count++)] = index + width;
  }
  if (x > 0) {
    neighbors[static_cast<std::size_t>(count++)] = index - 1;
  }
  if (x + 1 < width) {
    neighbors[static_cast<std::size_t>(count++)] = index + 1;
  }
}

std::vector<int32_t> largest_component_only(
    const std::vector<int32_t>& raw_mask,
    std::size_t depth,
    std::size_t height,
    std::size_t width) {
  const std::size_t voxel_count = raw_mask.size();
  std::vector<int32_t> visited(voxel_count, 0);
  std::vector<std::size_t> largest_component;
  std::vector<std::size_t> component;
  std::deque<std::size_t> queue;
  std::array<std::size_t, 6> neighbors{};

  for (std::size_t index = 0; index < voxel_count; ++index) {
    if (raw_mask[index] == 0 || visited[index] != 0) {
      continue;
    }

    component.clear();
    queue.clear();
    queue.push_back(index);
    visited[index] = 1;

    while (!queue.empty()) {
      const std::size_t current = queue.front();
      queue.pop_front();
      component.push_back(current);

      int neighbor_count = 0;
      push_neighbors_6(current, depth, height, width, neighbors, neighbor_count);
      for (int i = 0; i < neighbor_count; ++i) {
        const std::size_t next = neighbors[static_cast<std::size_t>(i)];
        if (raw_mask[next] == 0 || visited[next] != 0) {
          continue;
        }
        visited[next] = 1;
        queue.push_back(next);
      }
    }

    if (component.size() > largest_component.size()) {
      largest_component = component;
    }
  }

  std::vector<int32_t> cleaned(voxel_count, 0);
  for (std::size_t index : largest_component) {
    cleaned[index] = 1;
  }
  return cleaned;
}

std::vector<int32_t> fill_internal_holes(
    const std::vector<int32_t>& body_mask,
    std::size_t depth,
    std::size_t height,
    std::size_t width) {
  const std::size_t voxel_count = body_mask.size();
  std::vector<int32_t> outside(voxel_count, 0);
  std::deque<std::size_t> queue;
  std::array<std::size_t, 6> neighbors{};

  const auto seed_if_background = [&](std::size_t index) {
    if (body_mask[index] == 0 && outside[index] == 0) {
      outside[index] = 1;
      queue.push_back(index);
    }
  };

  for (std::size_t z = 0; z < depth; ++z) {
    for (std::size_t y = 0; y < height; ++y) {
      seed_if_background(flatten_index(z, y, 0, height, width));
      seed_if_background(flatten_index(z, y, width - 1, height, width));
    }
  }
  for (std::size_t z = 0; z < depth; ++z) {
    for (std::size_t x = 0; x < width; ++x) {
      seed_if_background(flatten_index(z, 0, x, height, width));
      seed_if_background(flatten_index(z, height - 1, x, height, width));
    }
  }
  for (std::size_t y = 0; y < height; ++y) {
    for (std::size_t x = 0; x < width; ++x) {
      seed_if_background(flatten_index(0, y, x, height, width));
      seed_if_background(flatten_index(depth - 1, y, x, height, width));
    }
  }

  while (!queue.empty()) {
    const std::size_t current = queue.front();
    queue.pop_front();

    int neighbor_count = 0;
    push_neighbors_6(current, depth, height, width, neighbors, neighbor_count);
    for (int i = 0; i < neighbor_count; ++i) {
      const std::size_t next = neighbors[static_cast<std::size_t>(i)];
      if (body_mask[next] != 0 || outside[next] != 0) {
        continue;
      }
      outside[next] = 1;
      queue.push_back(next);
    }
  }

  std::vector<int32_t> filled = body_mask;
  for (std::size_t index = 0; index < voxel_count; ++index) {
    if (body_mask[index] == 0 && outside[index] == 0) {
      filled[index] = 1;
    }
  }
  return filled;
}

}  // namespace

MaskResult build_and_clean_mask(
    const std::vector<double>& image,
    std::size_t t_count,
    std::size_t depth,
    std::size_t height,
    std::size_t width,
    const MaskOptions& options) {
  const std::size_t voxel_count = depth * height * width;
  if (image.size() != t_count * voxel_count) {
    throw std::runtime_error("Image size does not match supplied dimensions");
  }

  MaskResult result;
  result.raw_mask.assign(voxel_count, 0);

  for (std::size_t voxel = 0; voxel < voxel_count; ++voxel) {
    double max_value = 0.0;
    for (std::size_t t = 0; t < t_count; ++t) {
      max_value = std::max(max_value, image[t * voxel_count + voxel]);
    }
    result.raw_mask[voxel] = max_value >= options.threshold ? 1 : 0;
  }

  result.cleaned_mask = result.raw_mask;
  if (options.keep_largest_component) {
    result.cleaned_mask = largest_component_only(result.cleaned_mask, depth, height, width);
  }

  std::vector<int32_t> before_fill = result.cleaned_mask;
  if (options.fill_holes) {
    result.cleaned_mask = fill_internal_holes(result.cleaned_mask, depth, height, width);
  }

  result.removed_mask.assign(voxel_count, 0);
  result.filled_mask.assign(voxel_count, 0);
  for (std::size_t index = 0; index < voxel_count; ++index) {
    result.removed_mask[index] =
        result.raw_mask[index] != 0 && result.cleaned_mask[index] == 0 ? 1 : 0;
    result.filled_mask[index] =
        before_fill[index] == 0 && result.cleaned_mask[index] != 0 ? 1 : 0;
  }

  return result;
}

}  // namespace tcm
