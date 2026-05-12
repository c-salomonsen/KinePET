#pragma once

#include "solver.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tcm {

enum class ColumnMode {
  Shared,
  PerTac,
};

struct TacBatchData {
  std::vector<double> time_minutes;
  std::vector<double> tacs;
  std::vector<double> aif;
  std::vector<double> weights;
  std::size_t t_count = 0;
  std::size_t tac_count = 0;
  ColumnMode time_mode = ColumnMode::Shared;
  ColumnMode aif_mode = ColumnMode::Shared;
  ColumnMode weights_mode = ColumnMode::Shared;

  [[nodiscard]] bool has_weights() const noexcept { return !weights.empty(); }
};

struct TacBatchResult {
  std::vector<double> K1;
  std::vector<double> k2;
  std::vector<double> k3;
  std::vector<double> k4;
  std::vector<double> vB;
  std::vector<double> delay;
  std::vector<double> dispersion;
  std::vector<double> Ki;
  std::vector<double> rmse;
  std::vector<double> weighted_cost;
  std::vector<int32_t> iterations;
  std::vector<int32_t> status;
  std::size_t fitted_tacs = 0;
};

TacBatchData make_tac_batch_data(
    std::vector<double> time_minutes,
    std::size_t time_columns,
    std::vector<double> tacs,
    std::size_t t_count,
    std::size_t tac_count,
    std::vector<double> aif,
    std::size_t aif_columns,
    std::vector<double> weights = {},
    std::size_t weights_columns = 0);

TacBatchResult fit_tac_batch(
    const TacBatchData& batch,
    const SolverOptions& solver_options,
    int threads = 0);

}  // namespace tcm
