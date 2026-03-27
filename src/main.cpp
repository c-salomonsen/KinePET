#include "mask_utils.hpp"
#include "npy.hpp"
#include "post_filter.hpp"
#include "solver.hpp"
#include "tcm_model.hpp"

#include <algorithm>
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
  bool mask_keep_largest_component = true;
  bool mask_fill_holes = true;
  bool postfilter_status_median = true;
  int postfilter_radius = 1;
  int postfilter_min_reliable_neighbors = 3;
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
      << "  --mask-keep-largest-component 0|1  Keep only the main 3D body component (default: 1)\n"
      << "  --mask-fill-holes 0|1  Fill enclosed holes after component cleanup (default: 1)\n"
      << "  --postfilter-status-median 0|1  Apply selective 3D median filtering to status 2/4 voxels (default: 1)\n"
      << "  --postfilter-radius N  Radius for the masked 3D median window (default: 1, meaning 3x3x3)\n"
      << "  --postfilter-min-reliable-neighbors N  Minimum status 1/3 donors needed to replace a voxel (default: 3)\n"
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
    } else if (arg == "--mask-keep-largest-component") {
      options.mask_keep_largest_component = std::stoi(require_value(i, argc, argv)) != 0;
    } else if (arg == "--mask-fill-holes") {
      options.mask_fill_holes = std::stoi(require_value(i, argc, argv)) != 0;
    } else if (arg == "--postfilter-status-median") {
      options.postfilter_status_median = std::stoi(require_value(i, argc, argv)) != 0;
    } else if (arg == "--postfilter-radius") {
      options.postfilter_radius = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--postfilter-min-reliable-neighbors") {
      options.postfilter_min_reliable_neighbors = std::stoi(require_value(i, argc, argv));
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

void write_manifest(
    const fs::path& output_dir,
    const CliOptions& cli,
    std::size_t fitted,
    std::size_t total,
    std::size_t raw_mask_voxels,
    std::size_t cleaned_mask_voxels,
    const tcm::PostFilterResult& postfilter) {
  std::ofstream out(output_dir / "run.txt");
  out << "model=" << tcm::model_name(cli.model) << "\n";
  out << "fit_vb=" << (cli.fit_vb ? 1 : 0) << "\n";
  out << "fixed_vb=" << cli.fixed_vb << "\n";
  out << "mask_threshold=" << cli.mask_threshold << "\n";
  out << "mask_keep_largest_component=" << (cli.mask_keep_largest_component ? 1 : 0) << "\n";
  out << "mask_fill_holes=" << (cli.mask_fill_holes ? 1 : 0) << "\n";
  out << "raw_mask_voxels=" << raw_mask_voxels << "\n";
  out << "cleaned_mask_voxels=" << cleaned_mask_voxels << "\n";
  out << "postfilter_status_median=" << (cli.postfilter_status_median ? 1 : 0) << "\n";
  out << "postfilter_radius=" << cli.postfilter_radius << "\n";
  out << "postfilter_min_reliable_neighbors=" << cli.postfilter_min_reliable_neighbors << "\n";
  out << "postfilter_target_statuses=2,4\n";
  out << "postfilter_donor_statuses=1,3\n";
  out << "postfilter_targeted_voxels=" << postfilter.targeted_voxels << "\n";
  out << "postfilter_applied_voxels=" << postfilter.applied_voxels << "\n";
  out << "postfilter_unresolved_voxels=" << (postfilter.targeted_voxels - postfilter.applied_voxels)
      << "\n";
  out << "max_iterations=" << cli.max_iterations << "\n";
  out << "use_linear_warm_start=" << (cli.use_linear_warm_start ? 1 : 0) << "\n";
  out << "fitted_voxels=" << fitted << "\n";
  out << "total_voxels=" << total << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions cli = parse_args(argc, argv);
    if (cli.postfilter_radius < 1) {
      throw std::runtime_error("--postfilter-radius must be at least 1");
    }
    if (cli.postfilter_min_reliable_neighbors < 1) {
      throw std::runtime_error("--postfilter-min-reliable-neighbors must be at least 1");
    }
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

    tcm::MaskOptions mask_options;
    mask_options.threshold = cli.mask_threshold;
    mask_options.keep_largest_component = cli.mask_keep_largest_component;
    mask_options.fill_holes = cli.mask_fill_holes;
    const tcm::MaskResult mask_result =
        tcm::build_and_clean_mask(image, t_count, depth, height, width, mask_options);

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
    std::vector<int32_t> mask_map = mask_result.cleaned_mask;

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
        if (mask_map[static_cast<std::size_t>(voxel)] == 0) {
          continue;
        }

        for (std::size_t t = 0; t < t_count; ++t) {
          tac[t] = image[t * voxel_count + static_cast<std::size_t>(voxel)];
        }

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

    const std::vector<double> k1_rawfit = k1_map;
    const std::vector<double> k2_rawfit = k2_map;
    const std::vector<double> k3_rawfit = k3_map;
    const std::vector<double> k4_rawfit = k4_map;
    const std::vector<double> vb_rawfit = vb_map;
    const std::vector<double> ki_rawfit = ki_map;

    tcm::PostFilterOptions postfilter_options;
    postfilter_options.enable_status_median = cli.postfilter_status_median;
    postfilter_options.radius = cli.postfilter_radius;
    postfilter_options.min_reliable_neighbors = cli.postfilter_min_reliable_neighbors;
    const tcm::PostFilterResult postfilter = tcm::apply_selective_status_median_filter(
        cli.model,
        postfilter_options,
        mask_map,
        status_map,
        depth,
        height,
        width,
        k1_map,
        k2_map,
        k3_map,
        cli.model == tcm::ModelKind::Reversible2TCM ? &k4_map : nullptr,
        vb_map,
        ki_map);

    const std::vector<std::size_t> map_shape = {depth, height, width};
    tcm::save_npy_f64(cli.output_dir / "K1.npy", map_shape, k1_map);
    tcm::save_npy_f64(cli.output_dir / "k2.npy", map_shape, k2_map);
    tcm::save_npy_f64(cli.output_dir / "k3.npy", map_shape, k3_map);
    tcm::save_npy_f64(cli.output_dir / "K1_rawfit.npy", map_shape, k1_rawfit);
    tcm::save_npy_f64(cli.output_dir / "k2_rawfit.npy", map_shape, k2_rawfit);
    tcm::save_npy_f64(cli.output_dir / "k3_rawfit.npy", map_shape, k3_rawfit);
    if (cli.model == tcm::ModelKind::Reversible2TCM) {
      tcm::save_npy_f64(cli.output_dir / "k4.npy", map_shape, k4_map);
      tcm::save_npy_f64(cli.output_dir / "k4_rawfit.npy", map_shape, k4_rawfit);
    }
    tcm::save_npy_f64(cli.output_dir / "vB.npy", map_shape, vb_map);
    tcm::save_npy_f64(cli.output_dir / "Ki.npy", map_shape, ki_map);
    tcm::save_npy_f64(cli.output_dir / "vB_rawfit.npy", map_shape, vb_rawfit);
    tcm::save_npy_f64(cli.output_dir / "Ki_rawfit.npy", map_shape, ki_rawfit);
    tcm::save_npy_f64(cli.output_dir / "rmse.npy", map_shape, rmse_map);
    tcm::save_npy_i32(cli.output_dir / "iterations.npy", map_shape, iter_map);
    tcm::save_npy_i32(cli.output_dir / "status.npy", map_shape, status_map);
    tcm::save_npy_i32(cli.output_dir / "mask_raw.npy", map_shape, mask_result.raw_mask);
    tcm::save_npy_i32(cli.output_dir / "mask.npy", map_shape, mask_map);
    tcm::save_npy_i32(cli.output_dir / "mask_removed.npy", map_shape, mask_result.removed_mask);
    tcm::save_npy_i32(cli.output_dir / "mask_filled.npy", map_shape, mask_result.filled_mask);
    tcm::save_npy_i32(cli.output_dir / "postfilter_target.npy", map_shape, postfilter.target_mask);
    tcm::save_npy_i32(cli.output_dir / "postfilter_applied.npy", map_shape, postfilter.applied_mask);
    tcm::save_npy_i32(
        cli.output_dir / "postfilter_unresolved.npy", map_shape, postfilter.unresolved_mask);
    const std::size_t raw_mask_voxels = static_cast<std::size_t>(std::count(
        mask_result.raw_mask.begin(), mask_result.raw_mask.end(), 1));
    const std::size_t cleaned_mask_voxels =
        static_cast<std::size_t>(std::count(mask_map.begin(), mask_map.end(), 1));
    write_manifest(
        cli.output_dir,
        cli,
        fitted_voxels,
        voxel_count,
        raw_mask_voxels,
        cleaned_mask_voxels,
        postfilter);

    std::cout << "Finished " << fitted_voxels << " fitted voxels out of " << voxel_count << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
