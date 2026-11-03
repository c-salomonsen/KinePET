#include "tac_batch.hpp"

#include "tcm_model.hpp"
#include "frame_weights.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef TCM_HAS_OPENMP
#include <omp.h>
#endif

namespace tcm {
namespace {

void ensure_time_minutes(const std::vector<double>& time_minutes) {
  if (time_minutes.empty()) {
    throw std::runtime_error("time array must not be empty");
  }
}

std::size_t expected_column_payload(
    std::size_t t_count,
    std::size_t columns,
    const char* name) {
  if (columns == 0) {
    throw std::runtime_error(std::string(name) + " must have at least one column");
  }
  if (t_count == 0) {
    throw std::runtime_error("time dimension must be positive");
  }
  return t_count * columns;
}

void validate_payload_size(
    std::size_t payload_size,
    std::size_t expected_size,
    const char* name) {
  if (payload_size != expected_size) {
    throw std::runtime_error(
        std::string("Unexpected ") + name + " payload size: expected " +
        std::to_string(expected_size) + ", got " + std::to_string(payload_size));
  }
}

ColumnMode infer_mode(std::size_t columns, std::size_t tac_count, const char* name) {
  if (columns == 1) {
    return ColumnMode::Shared;
  }
  if (columns == tac_count) {
    return ColumnMode::PerTac;
  }
  throw std::runtime_error(
      std::string(name) + " second dimension must be 1 or match the number of TACs");
}

}  // namespace

TacBatchData make_tac_batch_data(
    std::vector<double> time_minutes,
    std::size_t time_columns,
    std::vector<double> tacs,
    std::size_t t_count,
    std::size_t tac_count,
    std::vector<double> aif,
    std::size_t aif_columns,
    std::vector<double> weights,
    std::size_t weights_columns) {
  if (t_count < 2) {
    throw std::runtime_error("At least two time points are required");
  }
  if (tac_count == 0) {
    throw std::runtime_error("At least one TAC is required");
  }
  const ColumnMode aif_mode = infer_mode(aif_columns, tac_count, "aif");
  const ColumnMode time_mode = infer_mode(time_columns, tac_count, "time");
  if (aif_mode == ColumnMode::PerTac && time_mode != ColumnMode::PerTac) {
    throw std::runtime_error("time must have shape (T, N) when aif has shape (T, N)");
  }
  if (aif_mode == ColumnMode::Shared && time_mode != ColumnMode::Shared) {
    throw std::runtime_error("time must be shared when aif is shared");
  }
  validate_payload_size(
      time_minutes.size(),
      expected_column_payload(t_count, time_columns, "time"),
      "time");
  validate_payload_size(tacs.size(), t_count * tac_count, "tacs");

  ensure_time_minutes(time_minutes);

  validate_payload_size(aif.size(), expected_column_payload(t_count, aif_columns, "aif"), "aif");

  ColumnMode weights_mode = ColumnMode::Shared;
  if (!weights.empty()) {
    weights_mode = infer_mode(weights_columns, tac_count, "weights");
    validate_payload_size(
        weights.size(),
        expected_column_payload(t_count, weights_columns, "weights"),
        "weights");
  }

  TacBatchData batch;
  batch.time_minutes = std::move(time_minutes);
  batch.tacs = std::move(tacs);
  batch.aif = std::move(aif);
  batch.weights = std::move(weights);
  batch.t_count = t_count;
  batch.tac_count = tac_count;
  batch.time_mode = time_mode;
  batch.aif_mode = aif_mode;
  batch.weights_mode = weights_mode;
  return batch;
}

TacBatchResult fit_tac_batch(
    const TacBatchData& batch,
    const SolverOptions& solver_options,
    int threads) {
  validate_payload_size(
      batch.time_minutes.size(),
      batch.t_count * (batch.time_mode == ColumnMode::Shared ? 1 : batch.tac_count),
      "time");
  if (batch.aif_mode == ColumnMode::PerTac && batch.time_mode != ColumnMode::PerTac) {
    throw std::runtime_error("Invalid TacBatchData: per-TAC AIFs require per-TAC time vectors");
  }
  if (batch.aif_mode == ColumnMode::Shared && batch.time_mode != ColumnMode::Shared) {
    throw std::runtime_error("Invalid TacBatchData: shared AIFs require shared time vectors");
  }
  validate_payload_size(batch.tacs.size(), batch.t_count * batch.tac_count, "tacs");
  validate_payload_size(
      batch.aif.size(),
      batch.t_count * (batch.aif_mode == ColumnMode::Shared ? 1 : batch.tac_count),
      "aif");
  if (batch.has_weights()) {
    validate_payload_size(
        batch.weights.size(),
        batch.t_count * (batch.weights_mode == ColumnMode::Shared ? 1 : batch.tac_count),
        "weights");
  }

#ifdef TCM_HAS_OPENMP
  if (threads > 0) {
    omp_set_num_threads(threads);
  }
#else
  (void) threads;
#endif

  TacBatchResult result;
  const std::size_t n = batch.tac_count;
  result.K1.assign(n, std::numeric_limits<double>::quiet_NaN());
  result.k2.assign(n, std::numeric_limits<double>::quiet_NaN());
  result.k3.assign(n, std::numeric_limits<double>::quiet_NaN());
  result.k4.assign(n, std::numeric_limits<double>::quiet_NaN());
  result.vB.assign(n, std::numeric_limits<double>::quiet_NaN());
  result.delay.assign(n, std::numeric_limits<double>::quiet_NaN());
  result.dispersion.assign(n, std::numeric_limits<double>::quiet_NaN());
  result.Ki.assign(n, std::numeric_limits<double>::quiet_NaN());
  result.rmse.assign(n, std::numeric_limits<double>::quiet_NaN());
  result.weighted_cost.assign(n, std::numeric_limits<double>::quiet_NaN());
  result.iterations.assign(n, 0);
  result.status.assign(n, static_cast<int32_t>(FitStatus::Failed));

  const std::vector<double> default_weights =
      batch.has_weights() || batch.time_mode == ColumnMode::PerTac
          ? std::vector<double>{}
          : default_frame_weights(batch.time_minutes);

#ifdef TCM_HAS_OPENMP
#pragma omp parallel
#endif
  {
    std::vector<double> time(batch.t_count, 0.0);
    std::vector<double> tac(batch.t_count, 0.0);
    std::vector<double> aif(batch.t_count, 0.0);
    std::vector<double> weights(batch.t_count, 0.0);
    std::size_t local_fitted = 0;

#ifdef TCM_HAS_OPENMP
#pragma omp for schedule(dynamic, 256)
#endif
    for (std::int64_t tac_index = 0; tac_index < static_cast<std::int64_t>(batch.tac_count); ++tac_index) {
      const std::size_t idx = static_cast<std::size_t>(tac_index);
      for (std::size_t t = 0; t < batch.t_count; ++t) {
        if (batch.time_mode == ColumnMode::Shared) {
          time[t] = batch.time_minutes[t];
        } else {
          time[t] = batch.time_minutes[t * batch.tac_count + idx];
        }
        tac[t] = batch.tacs[t * batch.tac_count + idx];

        if (batch.aif_mode == ColumnMode::Shared) {
          aif[t] = batch.aif[t];
        } else {
          aif[t] = batch.aif[t * batch.tac_count + idx];
        }

        if (!batch.has_weights()) {
          if (batch.time_mode == ColumnMode::Shared) {
            weights[t] = default_weights[t];
          }
        } else if (batch.weights_mode == ColumnMode::Shared) {
          weights[t] = batch.weights[t];
        } else {
          weights[t] = batch.weights[t * batch.tac_count + idx];
        }
      }
      if (!batch.has_weights() && batch.time_mode == ColumnMode::PerTac) {
        weights = default_frame_weights(time);
      }

      const FitResult fit = fit_tac(
          solver_options,
          time,
          aif,
          weights,
          tac);
      if (fit.status != FitStatus::InvalidInput) {
        result.K1[idx] = fit.params.K1;
        result.k2[idx] = fit.params.k2;
        result.k3[idx] = fit.params.k3;
        result.k4[idx] = fit.params.k4;
        result.vB[idx] = fit.params.vB;
        result.delay[idx] = fit.params.delay;
        result.dispersion[idx] = fit.params.dispersion;
        result.Ki[idx] = fit.ki;
        result.rmse[idx] = fit.rmse;
        result.weighted_cost[idx] = fit.weighted_cost;
        result.iterations[idx] = fit.iterations;
        result.status[idx] = static_cast<int32_t>(fit.status);
        ++local_fitted;
      }
    }

#ifdef TCM_HAS_OPENMP
#pragma omp atomic
#endif
    result.fitted_tacs += local_fitted;
  }

  if (!is_reversible(solver_options.model)) {
    std::fill(result.k4.begin(), result.k4.end(), 0.0);
  }

  if (!solver_options.aif.delay.fitted && std::abs(solver_options.aif.delay.fixed_value) <= 1e-12) {
    std::fill(result.delay.begin(), result.delay.end(), 0.0);
  }

  if (!solver_options.aif.dispersion.fitted && std::abs(solver_options.aif.dispersion.fixed_value) <= 1e-12) {
    std::fill(result.dispersion.begin(), result.dispersion.end(), 0.0);
  }

  return result;
}

}  // namespace tcm
