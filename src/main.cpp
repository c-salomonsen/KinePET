#include "npy.hpp"
#include "solver.hpp"
#include "tcm_model.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef TCM_HAS_OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

namespace {

struct CliOptions {
  fs::path input_dir;
  fs::path output_dir;
  tcm::ModelKind model = tcm::ModelKind::Reversible2TCM;
  bool fit_vb = true;
  double fixed_vb = 0.05;
  double mask_threshold = 0.05;
  int threads = 0;
  int max_iterations = 40;
  bool use_linear_warm_start = true;
  fs::path weights_file;
};

std::string require_value(int& index, int argc, char** argv) {
  if (index + 1 >= argc) {
    throw std::runtime_error(std::string("Missing value for argument: ") + argv[index]);
  }
  ++index;
  return argv[index];
}

void print_usage() {
  std::cout
      << "Usage: tcm_fit --input-dir DIR --output-dir DIR --model rev|irr [options]\n"
      << "Options:\n"
      << "  --fit-vb 0|1           Fit blood volume fraction (default: 1)\n"
      << "  --fixed-vb VALUE       Fixed blood volume when --fit-vb 0 (default: 0.05)\n"
      << "  --mask-threshold VAL   Skip voxels with max TAC below threshold (default: 0.05)\n"
      << "  --threads N            OpenMP threads when available\n"
      << "  --max-iter N           Maximum trust-region iterations (default: 40)\n"
      << "  --weights-file PATH    Optional .npy weights vector with length T\n"
      << "  --no-linear-warm-start Disable linear least-squares initialization\n";
}

CliOptions parse_args(int argc, char** argv) {
  CliOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input-dir") {
      options.input_dir = require_value(i, argc, argv);
    } else if (arg == "--output-dir") {
      options.output_dir = require_value(i, argc, argv);
    } else if (arg == "--model") {
      const std::string value = require_value(i, argc, argv);
      if (value == "rev") {
        options.model = tcm::ModelKind::Reversible2TCM;
      } else if (value == "irr") {
        options.model = tcm::ModelKind::Irreversible2TCM;
      } else {
        throw std::runtime_error("Model must be 'rev' or 'irr'");
      }
    } else if (arg == "--fit-vb") {
      options.fit_vb = std::stoi(require_value(i, argc, argv)) != 0;
    } else if (arg == "--fixed-vb") {
      options.fixed_vb = std::stod(require_value(i, argc, argv));
    } else if (arg == "--mask-threshold") {
      options.mask_threshold = std::stod(require_value(i, argc, argv));
    } else if (arg == "--threads") {
      options.threads = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--max-iter") {
      options.max_iterations = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--weights-file") {
      options.weights_file = require_value(i, argc, argv);
    } else if (arg == "--no-linear-warm-start") {
      options.use_linear_warm_start = false;
    } else if (arg == "--help" || arg == "-h") {
      print_usage();
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (options.input_dir.empty() || options.output_dir.empty()) {
    throw std::runtime_error("Both --input-dir and --output-dir are required");
  }

  return options;
}

std::vector<double> load_required_f64(const fs::path& path, std::vector<std::size_t>& shape) {
  if (!fs::exists(path)) {
    throw std::runtime_error("Missing file: " + path.string());
  }
  return tcm::load_npy_f64(path, shape);
}

void ensure_time_minutes(std::vector<double>& time_minutes) {
  const double max_time = *std::max_element(time_minutes.begin(), time_minutes.end());
  if (max_time > 60.0) {
    for (double& t : time_minutes) {
      t /= 60.0;
    }
  }
}

void write_manifest(const fs::path& output_dir, const CliOptions& cli, std::size_t fitted, std::size_t total) {
  std::ofstream out(output_dir / "run.txt");
  out << "model=" << tcm::model_name(cli.model) << "\n";
  out << "fit_vb=" << (cli.fit_vb ? 1 : 0) << "\n";
  out << "fixed_vb=" << cli.fixed_vb << "\n";
  out << "mask_threshold=" << cli.mask_threshold << "\n";
  out << "max_iterations=" << cli.max_iterations << "\n";
  out << "use_linear_warm_start=" << (cli.use_linear_warm_start ? 1 : 0) << "\n";
  out << "fitted_voxels=" << fitted << "\n";
  out << "total_voxels=" << total << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions cli = parse_args(argc, argv);
    fs::create_directories(cli.output_dir);

    std::vector<std::size_t> image_shape;
    std::vector<std::size_t> time_shape;
    std::vector<std::size_t> aif_shape;

    const std::vector<double> image =
        load_required_f64(cli.input_dir / "image.npy", image_shape);
    std::vector<double> time_minutes =
        load_required_f64(cli.input_dir / "time.npy", time_shape);
    std::vector<double> aif = load_required_f64(cli.input_dir / "aif.npy", aif_shape);

    if (image_shape.size() != 4) {
      throw std::runtime_error("image.npy must have shape (T, D, H, W)");
    }
    if (time_shape.size() != 1 || aif_shape.size() != 1) {
      throw std::runtime_error("time.npy and aif.npy must be 1D arrays");
    }

    ensure_time_minutes(time_minutes);

    const std::size_t t_count = image_shape[0];
    const std::size_t depth = image_shape[1];
    const std::size_t height = image_shape[2];
    const std::size_t width = image_shape[3];
    const std::size_t voxel_count = depth * height * width;

    if (time_minutes.size() != t_count || aif.size() != t_count) {
      throw std::runtime_error("Time dimension mismatch between image, time, and aif");
    }
    if (image.size() != t_count * voxel_count) {
      throw std::runtime_error("Unexpected image payload size");
    }

    std::vector<double> weights;
    if (!cli.weights_file.empty()) {
      std::vector<std::size_t> weight_shape;
      weights = tcm::load_npy_f64(cli.weights_file, weight_shape);
      if (weight_shape.size() != 1 || weights.size() != t_count) {
        throw std::runtime_error("weights.npy must be a 1D vector with length T");
      }
    } else {
      weights = tcm::default_frame_weights(time_minutes);
    }

    tcm::SolverOptions solver_options;
    solver_options.model = cli.model;
    solver_options.fit_vb = cli.fit_vb;
    solver_options.fixed_vb = cli.fixed_vb;
    solver_options.bounds = tcm::default_bounds(cli.model, cli.fit_vb, cli.fixed_vb);
    solver_options.max_iterations = cli.max_iterations;
    solver_options.use_linear_warm_start = cli.use_linear_warm_start;

#ifdef TCM_HAS_OPENMP
    if (cli.threads > 0) {
      omp_set_num_threads(cli.threads);
    }
#else
    (void)cli.threads;
#endif

    std::vector<double> k1_map(voxel_count, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> k2_map(voxel_count, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> k3_map(voxel_count, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> k4_map(
        voxel_count,
        cli.model == tcm::ModelKind::Reversible2TCM ? std::numeric_limits<double>::quiet_NaN() : 0.0);
    std::vector<double> vb_map(voxel_count, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> ki_map(voxel_count, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> rmse_map(voxel_count, std::numeric_limits<double>::quiet_NaN());
    std::vector<int32_t> iter_map(voxel_count, 0);
    std::vector<int32_t> status_map(voxel_count, static_cast<int32_t>(tcm::FitStatus::Failed));
    std::vector<int32_t> mask_map(voxel_count, 0);

    std::size_t fitted_voxels = 0;

#ifdef TCM_HAS_OPENMP
#pragma omp parallel
#endif
    {
      std::vector<double> tac(t_count, 0.0);
      std::size_t local_fitted = 0;

#ifdef TCM_HAS_OPENMP
#pragma omp for schedule(dynamic, 256)
#endif
      for (std::int64_t voxel = 0; voxel < static_cast<std::int64_t>(voxel_count); ++voxel) {
        double max_value = 0.0;
        for (std::size_t t = 0; t < t_count; ++t) {
          const double value = image[t * voxel_count + static_cast<std::size_t>(voxel)];
          tac[t] = value;
          max_value = std::max(max_value, value);
        }

        if (max_value < cli.mask_threshold) {
          continue;
        }

        mask_map[static_cast<std::size_t>(voxel)] = 1;
        const tcm::FitResult fit =
            tcm::fit_voxel(solver_options, time_minutes, aif, weights, tac);

        if (fit.status != tcm::FitStatus::InvalidInput) {
          k1_map[static_cast<std::size_t>(voxel)] = fit.params.K1;
          k2_map[static_cast<std::size_t>(voxel)] = fit.params.k2;
          k3_map[static_cast<std::size_t>(voxel)] = fit.params.k3;
          if (cli.model == tcm::ModelKind::Reversible2TCM) {
            k4_map[static_cast<std::size_t>(voxel)] = fit.params.k4;
          }
          vb_map[static_cast<std::size_t>(voxel)] = fit.params.vB;
          ki_map[static_cast<std::size_t>(voxel)] = fit.ki;
          rmse_map[static_cast<std::size_t>(voxel)] = fit.rmse;
          iter_map[static_cast<std::size_t>(voxel)] = fit.iterations;
          status_map[static_cast<std::size_t>(voxel)] = static_cast<int32_t>(fit.status);
          ++local_fitted;
        }
      }

#ifdef TCM_HAS_OPENMP
#pragma omp atomic
#endif
      fitted_voxels += local_fitted;
    }

    const std::vector<std::size_t> map_shape = {depth, height, width};
    tcm::save_npy_f64(cli.output_dir / "K1.npy", map_shape, k1_map);
    tcm::save_npy_f64(cli.output_dir / "k2.npy", map_shape, k2_map);
    tcm::save_npy_f64(cli.output_dir / "k3.npy", map_shape, k3_map);
    if (cli.model == tcm::ModelKind::Reversible2TCM) {
      tcm::save_npy_f64(cli.output_dir / "k4.npy", map_shape, k4_map);
    }
    tcm::save_npy_f64(cli.output_dir / "vB.npy", map_shape, vb_map);
    tcm::save_npy_f64(cli.output_dir / "Ki.npy", map_shape, ki_map);
    tcm::save_npy_f64(cli.output_dir / "rmse.npy", map_shape, rmse_map);
    tcm::save_npy_i32(cli.output_dir / "iterations.npy", map_shape, iter_map);
    tcm::save_npy_i32(cli.output_dir / "status.npy", map_shape, status_map);
    tcm::save_npy_i32(cli.output_dir / "mask.npy", map_shape, mask_map);
    write_manifest(cli.output_dir, cli, fitted_voxels, voxel_count);

    std::cout << "Finished " << fitted_voxels << " fitted voxels out of " << voxel_count << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
