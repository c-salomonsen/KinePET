#include "npy.hpp"
#include "solver.hpp"
#include "tcm_model.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
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
  bool fit_delay = false;
  double fixed_delay = 0.0;
  bool fit_dispersion = false;
  double fixed_dispersion = 0.0;
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
      << "Usage: tcm_fit_tacs --input-dir DIR --output-dir DIR --model rev|irr [options]\n"
      << "Inputs:\n"
      << "  tacs.npy  Shape (T, N)\n"
      << "  time.npy  Shape (T,)\n"
      << "  aif.npy   Shape (T,), (T,1), or (T,N)\n"
      << "Options:\n"
      << "  --fit-vb 0|1           Fit blood volume fraction (default: 1)\n"
      << "  --fixed-vb VALUE       Fixed blood volume when --fit-vb 0 (default: 0.05)\n"
      << "  --fit-delay 0|1        Fit AIF delay parameter Delta (default: 0)\n"
      << "  --fixed-delay VALUE    Fixed AIF delay in minutes when --fit-delay 0 (default: 0.0)\n"
      << "  --fit-dispersion 0|1   Fit AIF dispersion parameter tau (default: 0)\n"
      << "  --fixed-dispersion VALUE  Fixed AIF dispersion in minutes when --fit-dispersion 0 (default: 0.0)\n"
      << "  --threads N            OpenMP threads when available\n"
      << "  --max-iter N           Maximum trust-region iterations (default: 40)\n"
      << "  --weights-file PATH    Optional .npy weights array with shape (T,), (T,1), or (T,N)\n"
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
    } else if (arg == "--fit-delay") {
      options.fit_delay = std::stoi(require_value(i, argc, argv)) != 0;
    } else if (arg == "--fixed-delay") {
      options.fixed_delay = std::stod(require_value(i, argc, argv));
    } else if (arg == "--fit-dispersion") {
      options.fit_dispersion = std::stoi(require_value(i, argc, argv)) != 0;
    } else if (arg == "--fixed-dispersion") {
      options.fixed_dispersion = std::stod(require_value(i, argc, argv));
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
    bool shared_aif,
    bool shared_weights) {
  std::ofstream out(output_dir / "run.txt");
  out << "mode=tacs\n";
  out << "model=" << tcm::model_name(cli.model) << "\n";
  out << "fit_vb=" << (cli.fit_vb ? 1 : 0) << "\n";
  out << "fixed_vb=" << cli.fixed_vb << "\n";
  out << "fit_delay=" << (cli.fit_delay ? 1 : 0) << "\n";
  out << "fixed_delay=" << cli.fixed_delay << "\n";
  out << "fit_dispersion=" << (cli.fit_dispersion ? 1 : 0) << "\n";
  out << "fixed_dispersion=" << cli.fixed_dispersion << "\n";
  out << "shared_aif=" << (shared_aif ? 1 : 0) << "\n";
  out << "shared_weights=" << (shared_weights ? 1 : 0) << "\n";
  out << "max_iterations=" << cli.max_iterations << "\n";
  out << "use_linear_warm_start=" << (cli.use_linear_warm_start ? 1 : 0) << "\n";
  out << "fitted_tacs=" << fitted << "\n";
  out << "total_tacs=" << total << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions cli = parse_args(argc, argv);
    if (cli.fixed_dispersion < 0.0) {
      throw std::runtime_error("--fixed-dispersion must be non-negative");
    }
    fs::create_directories(cli.output_dir);

    std::vector<std::size_t> tac_shape;
    std::vector<std::size_t> time_shape;
    std::vector<std::size_t> aif_shape;

    const std::vector<double> tacs = load_required_f64(cli.input_dir / "tacs.npy", tac_shape);
    std::vector<double> time_minutes = load_required_f64(cli.input_dir / "time.npy", time_shape);
    const std::vector<double> aif_data = load_required_f64(cli.input_dir / "aif.npy", aif_shape);

    if (tac_shape.size() != 2) {
      throw std::runtime_error("tacs.npy must have shape (T, N)");
    }
    if (time_shape.size() != 1) {
      throw std::runtime_error("time.npy must have shape (T,)");
    }
    if (!(aif_shape.size() == 1 || aif_shape.size() == 2)) {
      throw std::runtime_error("aif.npy must have shape (T,), (T,1), or (T,N)");
    }

    ensure_time_minutes(time_minutes);

    const std::size_t t_count = tac_shape[0];
    const std::size_t tac_count = tac_shape[1];
    if (time_minutes.size() != t_count) {
      throw std::runtime_error("time.npy length must match tacs.npy first dimension");
    }
    if (tacs.size() != t_count * tac_count) {
      throw std::runtime_error("Unexpected tacs.npy payload size");
    }

    bool shared_aif = true;
    if (aif_shape.size() == 1) {
      if (aif_data.size() != t_count) {
        throw std::runtime_error("aif.npy with shape (T,) must match time.npy length");
      }
    } else {
      if (aif_shape[0] != t_count) {
        throw std::runtime_error("aif.npy first dimension must match tacs.npy first dimension");
      }
      if (aif_shape[1] == 1) {
        shared_aif = true;
      } else if (aif_shape[1] == tac_count) {
        shared_aif = false;
      } else {
        throw std::runtime_error("aif.npy second dimension must be 1 or match the number of TACs");
      }
      if (aif_data.size() != aif_shape[0] * aif_shape[1]) {
        throw std::runtime_error("Unexpected aif.npy payload size");
      }
    }

    std::vector<double> weights_data;
    std::vector<std::size_t> weight_shape;
    bool shared_weights = true;
    if (!cli.weights_file.empty()) {
      weights_data = tcm::load_npy_f64(cli.weights_file, weight_shape);
      if (!(weight_shape.size() == 1 || weight_shape.size() == 2)) {
        throw std::runtime_error("weights.npy must have shape (T,), (T,1), or (T,N)");
      }
      if (weight_shape.size() == 1) {
        if (weights_data.size() != t_count) {
          throw std::runtime_error("weights.npy with shape (T,) must match time.npy length");
        }
      } else {
        if (weight_shape[0] != t_count) {
          throw std::runtime_error("weights.npy first dimension must match tacs.npy first dimension");
        }
        if (weight_shape[1] == 1) {
          shared_weights = true;
        } else if (weight_shape[1] == tac_count) {
          shared_weights = false;
        } else {
          throw std::runtime_error("weights.npy second dimension must be 1 or match the number of TACs");
        }
        if (weights_data.size() != weight_shape[0] * weight_shape[1]) {
          throw std::runtime_error("Unexpected weights.npy payload size");
        }
      }
    }

    const std::vector<double> default_weights =
        cli.weights_file.empty() ? tcm::default_frame_weights(time_minutes) : std::vector<double>{};

    tcm::SolverOptions solver_options;
    solver_options.model = cli.model;
    solver_options.fit_vb = cli.fit_vb;
    solver_options.fixed_vb = cli.fixed_vb;
    solver_options.fit_delay = cli.fit_delay;
    solver_options.fixed_delay = cli.fixed_delay;
    solver_options.fit_dispersion = cli.fit_dispersion;
    solver_options.fixed_dispersion = cli.fixed_dispersion;
    solver_options.bounds = tcm::default_bounds(
        cli.model,
        cli.fit_vb,
        cli.fixed_vb,
        cli.fit_delay,
        cli.fixed_delay,
        cli.fit_dispersion,
        cli.fixed_dispersion);
    solver_options.max_iterations = cli.max_iterations;
    solver_options.use_linear_warm_start = cli.use_linear_warm_start;

#ifdef TCM_HAS_OPENMP
    if (cli.threads > 0) {
      omp_set_num_threads(cli.threads);
    }
#else
    (void)cli.threads;
#endif

    std::vector<double> k1_map(tac_count, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> k2_map(tac_count, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> k3_map(tac_count, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> k4_map(
        tac_count,
        cli.model == tcm::ModelKind::Reversible2TCM ? std::numeric_limits<double>::quiet_NaN() : 0.0);
    std::vector<double> vb_map(tac_count, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> delay_map(tac_count, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> dispersion_map(tac_count, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> ki_map(tac_count, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> rmse_map(tac_count, std::numeric_limits<double>::quiet_NaN());
    std::vector<int32_t> iter_map(tac_count, 0);
    std::vector<int32_t> status_map(tac_count, static_cast<int32_t>(tcm::FitStatus::Failed));

    std::size_t fitted_tacs = 0;

#ifdef TCM_HAS_OPENMP
#pragma omp parallel
#endif
    {
      std::vector<double> tac(t_count, 0.0);
      std::vector<double> aif(t_count, 0.0);
      std::vector<double> weights(t_count, 0.0);
      std::size_t local_fitted = 0;

#ifdef TCM_HAS_OPENMP
#pragma omp for schedule(dynamic, 256)
#endif
      for (std::int64_t tac_index = 0; tac_index < static_cast<std::int64_t>(tac_count); ++tac_index) {
        const std::size_t idx = static_cast<std::size_t>(tac_index);
        for (std::size_t t = 0; t < t_count; ++t) {
          tac[t] = tacs[t * tac_count + idx];
          if (shared_aif) {
            if (aif_shape.size() == 1) {
              aif[t] = aif_data[t];
            } else {
              aif[t] = aif_data[t];
            }
          } else {
            aif[t] = aif_data[t * tac_count + idx];
          }

          if (cli.weights_file.empty()) {
            weights[t] = default_weights[t];
          } else if (shared_weights) {
            if (weight_shape.size() == 1) {
              weights[t] = weights_data[t];
            } else {
              weights[t] = weights_data[t];
            }
          } else {
            weights[t] = weights_data[t * tac_count + idx];
          }
        }

        const tcm::FitResult fit = tcm::fit_voxel(solver_options, time_minutes, aif, weights, tac);
        if (fit.status != tcm::FitStatus::InvalidInput) {
          k1_map[idx] = fit.params.K1;
          k2_map[idx] = fit.params.k2;
          k3_map[idx] = fit.params.k3;
          if (cli.model == tcm::ModelKind::Reversible2TCM) {
            k4_map[idx] = fit.params.k4;
          }
          vb_map[idx] = fit.params.vB;
          delay_map[idx] = fit.params.delay;
          dispersion_map[idx] = fit.params.dispersion;
          ki_map[idx] = fit.ki;
          rmse_map[idx] = fit.rmse;
          iter_map[idx] = fit.iterations;
          status_map[idx] = static_cast<int32_t>(fit.status);
          ++local_fitted;
        }
      }

#ifdef TCM_HAS_OPENMP
#pragma omp atomic
#endif
      fitted_tacs += local_fitted;
    }

    const std::vector<std::size_t> out_shape = {tac_count};
    tcm::save_npy_f64(cli.output_dir / "K1.npy", out_shape, k1_map);
    tcm::save_npy_f64(cli.output_dir / "k2.npy", out_shape, k2_map);
    tcm::save_npy_f64(cli.output_dir / "k3.npy", out_shape, k3_map);
    if (cli.model == tcm::ModelKind::Reversible2TCM) {
      tcm::save_npy_f64(cli.output_dir / "k4.npy", out_shape, k4_map);
    }
    tcm::save_npy_f64(cli.output_dir / "vB.npy", out_shape, vb_map);
    tcm::save_npy_f64(cli.output_dir / "Ki.npy", out_shape, ki_map);
    if (cli.fit_delay || std::abs(cli.fixed_delay) > 1e-12) {
      tcm::save_npy_f64(cli.output_dir / "delay.npy", out_shape, delay_map);
    }
    if (cli.fit_dispersion || std::abs(cli.fixed_dispersion) > 1e-12) {
      tcm::save_npy_f64(cli.output_dir / "dispersion.npy", out_shape, dispersion_map);
    }
    tcm::save_npy_f64(cli.output_dir / "rmse.npy", out_shape, rmse_map);
    tcm::save_npy_i32(cli.output_dir / "iterations.npy", out_shape, iter_map);
    tcm::save_npy_i32(cli.output_dir / "status.npy", out_shape, status_map);
    write_manifest(cli.output_dir, cli, fitted_tacs, tac_count, shared_aif, shared_weights);

    std::cout << "Finished " << fitted_tacs << " fitted TACs out of " << tac_count << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
