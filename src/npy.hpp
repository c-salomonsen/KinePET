#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace tcm {

struct NpyArray {
  std::vector<std::size_t> shape;
  std::string descr;
  bool fortran_order = false;
  std::vector<std::byte> raw;

  [[nodiscard]] std::size_t numel() const;
};

NpyArray load_npy(const std::filesystem::path& path);
std::vector<double> load_npy_f64(const std::filesystem::path& path, std::vector<std::size_t>& shape);
std::vector<int32_t> load_npy_i32(
    const std::filesystem::path& path,
    std::vector<std::size_t>& shape);
void save_npy_f64(
    const std::filesystem::path& path,
    const std::vector<std::size_t>& shape,
    const std::vector<double>& data);
void save_npy_i32(
    const std::filesystem::path& path,
    const std::vector<std::size_t>& shape,
    const std::vector<int32_t>& data);

}  // namespace tcm
