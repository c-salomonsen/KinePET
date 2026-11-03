#include "npy.hpp"
#include "solver.hpp"
#include "tac_batch.hpp"
#include "tcm_model.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct CliOptions {
  fs::path input_dir;
  fs::path output_dir;
  std::string model = "rev";
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
      << "  time.npy  Shape (T,), (T,1), or (T,N), in minutes. Use (T,N) with per-TAC AIFs.\n"
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
        options.model = "rev";
      } else if (value == "irr") {
        options.model = "irr";
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

void write_manifest(
    const fs::path& output_dir,
    const CliOptions& cli,
    std::size_t fitted,
    std::size_t total,
    bool shared_time,
    bool shared_aif,
    bool shared_weights) {
  std::ofstream out(output_dir / "run.txt");
  out << "mode=tacs\n";
  out << "model=" << tcm::model_name(tcm::make_model(cli.model)) << "\n";
  out << "fit_vb=" << (cli.fit_vb ? 1 : 0) << "\n";
  out << "fixed_vb=" << cli.fixed_vb << "\n";
  out << "fit_delay=" << (cli.fit_delay ? 1 : 0) << "\n";
  out << "fixed_delay=" << cli.fixed_delay << "\n";
  out << "fit_dispersion=" << (cli.fit_dispersion ? 1 : 0) << "\n";
  out << "fixed_dispersion=" << cli.fixed_dispersion << "\n";
  out << "shared_time=" << (shared_time ? 1 : 0) << "\n";
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

    std::vector<double> tacs = load_required_f64(cli.input_dir / "tacs.npy", tac_shape);
    std::vector<double> time_minutes = load_required_f64(cli.input_dir / "time.npy", time_shape);
    std::vector<double> aif_data = load_required_f64(cli.input_dir / "aif.npy", aif_shape);

    if (tac_shape.size() != 2) {
      throw std::runtime_error("tacs.npy must have shape (T, N)");
    }
    if (!(time_shape.size() == 1 || time_shape.size() == 2)) {
      throw std::runtime_error("time.npy must have shape (T,), (T,1), or (T,N)");
    }
    if (!(aif_shape.size() == 1 || aif_shape.size() == 2)) {
      throw std::runtime_error("aif.npy must have shape (T,), (T,1), or (T,N)");
    }

    const std::size_t t_count = tac_shape[0];
    const std::size_t tac_count = tac_shape[1];
    std::size_t time_columns = 1;
    if (time_shape.size() == 1) {
      if (time_minutes.size() != t_count) {
        throw std::runtime_error("time.npy with shape (T,) must match tacs.npy first dimension");
      }
    } else {
      if (time_shape[0] != t_count) {
        throw std::runtime_error("time.npy first dimension must match tacs.npy first dimension");
      }
      time_columns = time_shape[1];
      if (time_minutes.size() != time_shape[0] * time_shape[1]) {
        throw std::runtime_error("Unexpected time.npy payload size");
      }
    }
    if (tacs.size() != t_count * tac_count) {
      throw std::runtime_error("Unexpected tacs.npy payload size");
    }

    std::size_t aif_columns = 1;
    if (aif_shape.size() == 1) {
      if (aif_data.size() != t_count) {
        throw std::runtime_error("aif.npy with shape (T,) must match time.npy length");
      }
    } else {
      if (aif_shape[0] != t_count) {
        throw std::runtime_error("aif.npy first dimension must match tacs.npy first dimension");
      }
      aif_columns = aif_shape[1];
      if (aif_data.size() != aif_shape[0] * aif_shape[1]) {
        throw std::runtime_error("Unexpected aif.npy payload size");
      }
    }

    std::vector<double> weights_data;
    std::vector<std::size_t> weight_shape;
    std::size_t weights_columns = 0;
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
        weights_columns = weight_shape[1];
        if (weights_data.size() != weight_shape[0] * weight_shape[1]) {
          throw std::runtime_error("Unexpected weights.npy payload size");
        }
      }
    }

    tcm::SolverOptions solver_options;
    solver_options.model = tcm::make_model(cli.model, tcm::blood_fraction_setting(cli.fit_vb, cli.fixed_vb));
    solver_options.aif.delay.fitted = cli.fit_delay;
    solver_options.aif.delay.fixed_value = cli.fixed_delay;
    solver_options.aif.dispersion.fitted = cli.fit_dispersion;
    solver_options.aif.dispersion.fixed_value = cli.fixed_dispersion;
    solver_options.bounds = tcm::default_bounds(
        solver_options.model, solver_options.aif);
    solver_options.max_iterations = cli.max_iterations;
    solver_options.use_linear_warm_start = cli.use_linear_warm_start;
    const tcm::TacBatchData batch = tcm::make_tac_batch_data(
        std::move(time_minutes),
        time_columns,
        std::move(tacs),
        t_count,
        tac_count,
        std::move(aif_data),
        aif_columns,
        std::move(weights_data),
        weight_shape.empty() ? 0 : (weight_shape.size() == 1 ? 1 : weights_columns));
    const tcm::TacBatchResult result = tcm::fit_tac_batch(batch, solver_options, cli.threads);

    const std::vector<std::size_t> out_shape = {tac_count};
    tcm::save_npy_f64(cli.output_dir / "K1.npy", out_shape, result.K1);
    tcm::save_npy_f64(cli.output_dir / "k2.npy", out_shape, result.k2);
    tcm::save_npy_f64(cli.output_dir / "k3.npy", out_shape, result.k3);
    if (cli.model == "rev") {
      tcm::save_npy_f64(cli.output_dir / "k4.npy", out_shape, result.k4);
    }
    tcm::save_npy_f64(cli.output_dir / "vB.npy", out_shape, result.vB);
    tcm::save_npy_f64(cli.output_dir / "Ki.npy", out_shape, result.Ki);
    if (cli.fit_delay || std::abs(cli.fixed_delay) > 1e-12) {
      tcm::save_npy_f64(cli.output_dir / "delay.npy", out_shape, result.delay);
    }
    if (cli.fit_dispersion || std::abs(cli.fixed_dispersion) > 1e-12) {
      tcm::save_npy_f64(cli.output_dir / "dispersion.npy", out_shape, result.dispersion);
    }
    tcm::save_npy_f64(cli.output_dir / "rmse.npy", out_shape, result.rmse);
    tcm::save_npy_i32(cli.output_dir / "iterations.npy", out_shape, result.iterations);
    tcm::save_npy_i32(cli.output_dir / "status.npy", out_shape, result.status);
    write_manifest(
        cli.output_dir,
        cli,
        result.fitted_tacs,
        tac_count,
        batch.time_mode == tcm::ColumnMode::Shared,
        batch.aif_mode == tcm::ColumnMode::Shared,
        !batch.has_weights() || batch.weights_mode == tcm::ColumnMode::Shared);

    std::cout << "Finished " << result.fitted_tacs << " fitted TACs out of " << tac_count << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
