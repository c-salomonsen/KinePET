#include "npy.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tcm {
namespace {

constexpr std::array<char, 6> kMagic = {'\x93', 'N', 'U', 'M', 'P', 'Y'};

std::size_t product(const std::vector<std::size_t>& shape) {
  std::size_t value = 1;
  for (std::size_t dim : shape) {
    value *= dim;
  }
  return value;
}

std::string extract_between(
    const std::string& header,
    const std::string& key,
    char begin,
    char end) {
  const std::size_t key_pos = header.find(key);
  if (key_pos == std::string::npos) {
    throw std::runtime_error("Missing NPY header key: " + key);
  }
  const std::size_t begin_pos = header.find(begin, key_pos + key.size());
  const std::size_t end_pos = header.find(end, begin_pos + 1);
  if (begin_pos == std::string::npos || end_pos == std::string::npos) {
    throw std::runtime_error("Malformed NPY header for key: " + key);
  }
  return header.substr(begin_pos + 1, end_pos - begin_pos - 1);
}

std::vector<std::size_t> parse_shape(const std::string& header) {
  const std::size_t key_pos = header.find("'shape'");
  if (key_pos == std::string::npos) {
    throw std::runtime_error("Missing shape in NPY header");
  }
  const std::size_t left = header.find('(', key_pos);
  const std::size_t right = header.find(')', left + 1);
  if (left == std::string::npos || right == std::string::npos) {
    throw std::runtime_error("Malformed shape in NPY header");
  }

  std::string content = header.substr(left + 1, right - left - 1);
  std::vector<std::size_t> shape;
  std::stringstream ss(content);
  std::string token;
  while (std::getline(ss, token, ',')) {
    std::stringstream trimmed(token);
    std::size_t dim = 0;
    if (trimmed >> dim) {
      shape.push_back(dim);
    }
  }

  if (shape.empty()) {
    throw std::runtime_error("Scalar arrays are not supported");
  }
  return shape;
}

template <typename T>
void save_npy_impl(
    const std::filesystem::path& path,
    const std::vector<std::size_t>& shape,
    const std::vector<T>& data,
    const std::string& descr) {
  if (product(shape) != data.size()) {
    throw std::runtime_error("NPY save shape does not match data size");
  }

  std::ostringstream shape_stream;
  shape_stream << "(";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    shape_stream << shape[i];
    if (shape.size() == 1 || i + 1 < shape.size()) {
      shape_stream << ",";
    }
    if (i + 1 < shape.size()) {
      shape_stream << " ";
    }
  }
  shape_stream << ")";

  std::string header =
      "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " +
      shape_stream.str() + ", }";

  // Header must end on a 16-byte boundary including magic, version, and header length.
  while ((10 + header.size() + 1) % 16 != 0) {
    header.push_back(' ');
  }
  header.push_back('\n');

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Failed to open file for writing: " + path.string());
  }

  out.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
  const std::uint8_t version[2] = {1, 0};
  out.write(reinterpret_cast<const char*>(version), 2);
  const std::uint16_t header_len = static_cast<std::uint16_t>(header.size());
  out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
}

template <typename T>
std::vector<T> convert_raw(const NpyArray& array, const std::string& expected_descr) {
  if (array.descr != expected_descr) {
    throw std::runtime_error(
        "Unsupported dtype: expected " + expected_descr + " but found " + array.descr);
  }
  if (array.fortran_order) {
    throw std::runtime_error("Fortran-order NPY arrays are not supported");
  }
  const std::size_t expected_bytes = array.numel() * sizeof(T);
  if (array.raw.size() != expected_bytes) {
    throw std::runtime_error("NPY payload size mismatch");
  }
  std::vector<T> values(array.numel());
  std::memcpy(values.data(), array.raw.data(), expected_bytes);
  return values;
}

}  // namespace

std::size_t NpyArray::numel() const {
  return product(shape);
}

NpyArray load_npy(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Failed to open file: " + path.string());
  }

  std::array<char, 6> magic{};
  in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (magic != kMagic) {
    throw std::runtime_error("Invalid NPY magic header: " + path.string());
  }

  std::uint8_t major = 0;
  std::uint8_t minor = 0;
  in.read(reinterpret_cast<char*>(&major), 1);
  in.read(reinterpret_cast<char*>(&minor), 1);

  std::uint32_t header_len = 0;
  if (major == 1) {
    std::uint16_t len16 = 0;
    in.read(reinterpret_cast<char*>(&len16), sizeof(len16));
    header_len = len16;
  } else if (major == 2) {
    in.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
  } else {
    throw std::runtime_error("Unsupported NPY version");
  }

  std::string header(header_len, '\0');
  in.read(header.data(), static_cast<std::streamsize>(header_len));

  NpyArray array;
  array.descr = extract_between(header, "'descr'", '\'', '\'');
  array.fortran_order = header.find("'fortran_order': True") != std::string::npos;
  array.shape = parse_shape(header);

  const std::size_t remaining = product(array.shape) *
      (array.descr == "<f8" ? sizeof(double) : array.descr == "<i4" ? sizeof(std::int32_t) : 0);
  if (remaining == 0) {
    throw std::runtime_error("Only <f8 and <i4 NPY arrays are supported");
  }

  array.raw.resize(remaining);
  in.read(reinterpret_cast<char*>(array.raw.data()), static_cast<std::streamsize>(remaining));
  if (!in) {
    throw std::runtime_error("Failed to read NPY payload: " + path.string());
  }

  return array;
}

std::vector<double> load_npy_f64(
    const std::filesystem::path& path,
    std::vector<std::size_t>& shape) {
  const NpyArray array = load_npy(path);
  shape = array.shape;
  return convert_raw<double>(array, "<f8");
}

std::vector<int32_t> load_npy_i32(
    const std::filesystem::path& path,
    std::vector<std::size_t>& shape) {
  const NpyArray array = load_npy(path);
  shape = array.shape;
  return convert_raw<int32_t>(array, "<i4");
}

void save_npy_f64(
    const std::filesystem::path& path,
    const std::vector<std::size_t>& shape,
    const std::vector<double>& data) {
  save_npy_impl(path, shape, data, "<f8");
}

void save_npy_i32(
    const std::filesystem::path& path,
    const std::vector<std::size_t>& shape,
    const std::vector<int32_t>& data) {
  save_npy_impl(path, shape, data, "<i4");
}

}  // namespace tcm
