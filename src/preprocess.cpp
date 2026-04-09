#include "preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

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

std::vector<double> gaussian_kernel(int radius, double sigma) {
  if (radius < 1 || sigma <= 0.0) {
    return {1.0};
  }
  const int size = 2 * radius + 1;
  std::vector<double> kernel(static_cast<std::size_t>(size), 0.0);
  double sum = 0.0;
  for (int i = -radius; i <= radius; ++i) {
    const double value = std::exp(-0.5 * static_cast<double>(i * i) / (sigma * sigma));
    kernel[static_cast<std::size_t>(i + radius)] = value;
    sum += value;
  }
  if (sum > 0.0) {
    for (double& value : kernel) {
      value /= sum;
    }
  }
  return kernel;
}

std::size_t clamp_index(std::ptrdiff_t value, std::size_t limit) {
  if (value < 0) {
    return 0;
  }
  const std::size_t cast = static_cast<std::size_t>(value);
  return std::min(limit - 1, cast);
}

void apply_spatial_gaussian_inplace(
    std::vector<double>& image,
    std::size_t t_count,
    std::size_t depth,
    std::size_t height,
    std::size_t width,
    int radius,
    double sigma) {
  if (radius < 1 || sigma <= 0.0) {
    return;
  }

  const std::vector<double> kernel = gaussian_kernel(radius, sigma);
  const std::size_t voxel_count = depth * height * width;
  std::vector<double> frame(voxel_count, 0.0);
  std::vector<double> temp(voxel_count, 0.0);

  for (std::size_t t = 0; t < t_count; ++t) {
    const std::size_t offset = t * voxel_count;
    std::copy(image.begin() + static_cast<std::ptrdiff_t>(offset),
              image.begin() + static_cast<std::ptrdiff_t>(offset + voxel_count),
              frame.begin());

    for (std::size_t z = 0; z < depth; ++z) {
      for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
          double value = 0.0;
          for (int dx = -radius; dx <= radius; ++dx) {
            const std::size_t xx = clamp_index(static_cast<std::ptrdiff_t>(x) + dx, width);
            value += kernel[static_cast<std::size_t>(dx + radius)] *
                frame[flatten_index(z, y, xx, height, width)];
          }
          temp[flatten_index(z, y, x, height, width)] = value;
        }
      }
    }
    frame.swap(temp);

    for (std::size_t z = 0; z < depth; ++z) {
      for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
          double value = 0.0;
          for (int dy = -radius; dy <= radius; ++dy) {
            const std::size_t yy = clamp_index(static_cast<std::ptrdiff_t>(y) + dy, height);
            value += kernel[static_cast<std::size_t>(dy + radius)] *
                frame[flatten_index(z, yy, x, height, width)];
          }
          temp[flatten_index(z, y, x, height, width)] = value;
        }
      }
    }
    frame.swap(temp);

    for (std::size_t z = 0; z < depth; ++z) {
      for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
          double value = 0.0;
          for (int dz = -radius; dz <= radius; ++dz) {
            const std::size_t zz = clamp_index(static_cast<std::ptrdiff_t>(z) + dz, depth);
            value += kernel[static_cast<std::size_t>(dz + radius)] *
                frame[flatten_index(zz, y, x, height, width)];
          }
          temp[flatten_index(z, y, x, height, width)] = value;
        }
      }
    }

    std::copy(temp.begin(), temp.end(), image.begin() + static_cast<std::ptrdiff_t>(offset));
  }
}

double median_of_window(std::vector<double>& window) {
  std::sort(window.begin(), window.end());
  const std::size_t mid = window.size() / 2;
  if ((window.size() % 2) != 0) {
    return window[mid];
  }
  return 0.5 * (window[mid - 1] + window[mid]);
}

void apply_temporal_filter_inplace(
    std::vector<double>& image,
    std::size_t t_count,
    std::size_t depth,
    std::size_t height,
    std::size_t width,
    TemporalFilterKind kind,
    int radius,
    double sigma) {
  if (kind == TemporalFilterKind::None || radius < 1) {
    return;
  }

  const std::size_t voxel_count = depth * height * width;
  std::vector<double> tac(t_count, 0.0);
  std::vector<double> filtered(t_count, 0.0);
  std::vector<double> window;
  const std::vector<double> kernel =
      kind == TemporalFilterKind::Gaussian ? gaussian_kernel(radius, sigma) : std::vector<double>{};

  for (std::size_t voxel = 0; voxel < voxel_count; ++voxel) {
    for (std::size_t t = 0; t < t_count; ++t) {
      tac[t] = image[t * voxel_count + voxel];
    }

    for (std::size_t t = 0; t < t_count; ++t) {
      if (kind == TemporalFilterKind::Median) {
        window.clear();
        for (int dt = -radius; dt <= radius; ++dt) {
          const std::size_t tt = clamp_index(static_cast<std::ptrdiff_t>(t) + dt, t_count);
          window.push_back(tac[tt]);
        }
        filtered[t] = median_of_window(window);
      } else {
        double value = 0.0;
        for (int dt = -radius; dt <= radius; ++dt) {
          const std::size_t tt = clamp_index(static_cast<std::ptrdiff_t>(t) + dt, t_count);
          value += kernel[static_cast<std::size_t>(dt + radius)] * tac[tt];
        }
        filtered[t] = value;
      }
    }

    for (std::size_t t = 0; t < t_count; ++t) {
      image[t * voxel_count + voxel] = filtered[t];
    }
  }
}

}  // namespace

std::string temporal_filter_name(TemporalFilterKind kind) {
  switch (kind) {
    case TemporalFilterKind::None:
      return "none";
    case TemporalFilterKind::Median:
      return "median";
    case TemporalFilterKind::Gaussian:
      return "gaussian";
  }
  throw std::runtime_error("Unknown temporal filter kind");
}

std::vector<double> apply_prefit_preprocessing(
    const std::vector<double>& image,
    std::size_t t_count,
    std::size_t depth,
    std::size_t height,
    std::size_t width,
    const PreprocessingOptions& options) {
  std::vector<double> processed = image;
  if (options.spatial_gaussian_enable) {
    apply_spatial_gaussian_inplace(
        processed,
        t_count,
        depth,
        height,
        width,
        options.spatial_gaussian_radius,
        options.spatial_gaussian_sigma);
  }
  apply_temporal_filter_inplace(
      processed,
      t_count,
      depth,
      height,
      width,
      options.temporal_filter,
      options.temporal_filter_radius,
      options.temporal_gaussian_sigma);
  return processed;
}

}  // namespace tcm
