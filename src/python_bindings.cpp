#include "solver.hpp"
#include "tac_batch.hpp"
#include "tcm_model.hpp"
#include "frame_weights.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using CpuArray = nb::ndarray<const double, nb::device::cpu>;
using CpuVector = nb::ndarray<const double, nb::device::cpu, nb::ndim<1>>;

template <typename Array>
std::size_t element_count(const Array& array) {
  std::size_t total = 1;
  for (std::size_t axis = 0; axis < array.ndim(); ++axis) {
    total *= array.shape(axis);
  }
  return total;
}

template <typename Array>
std::vector<double> copy_array(const Array& array) {
  const double* begin = static_cast<const double*>(array.data());
  return std::vector<double>(begin, begin + element_count(array));
}

std::vector<double> copy_time_minutes(const CpuVector& time) {
  return copy_array(time);
}

std::size_t infer_columns(const CpuArray& array, std::size_t expected_rows, const char* name) {
  if (array.ndim() == 1) {
    if (array.shape(0) != expected_rows) {
      throw std::runtime_error(std::string(name) + " length must match the time vector");
    }
    return 1;
  }
  if (array.ndim() == 2) {
    if (array.shape(0) != expected_rows) {
      throw std::runtime_error(
          std::string(name) + " first dimension must match the time vector length");
    }
    return array.shape(1);
  }
  throw std::runtime_error(std::string(name) + " must be a 1D or 2D array");
}

tcm::SolverOptions make_solver_options(
    const std::string& model_name,
    bool fit_vb,
    double fixed_vb,
    bool fit_delay,
    double fixed_delay,
    bool fit_dispersion,
    double fixed_dispersion,
    int max_iterations,
    double gradient_tolerance,
    double step_tolerance,
    double cost_tolerance,
    double fd_step,
    double lambda_initial,
    bool use_linear_warm_start) {
  if (fixed_dispersion < 0.0) {
    throw std::runtime_error("fixed_dispersion must be non-negative");
  }

  tcm::SolverOptions options;
  options.model = tcm::make_model(model_name, tcm::blood_fraction_setting(fit_vb, fixed_vb));
  options.aif.delay.fitted = fit_delay;
  options.aif.delay.fixed_value = fixed_delay;
  options.aif.dispersion.fitted = fit_dispersion;
  options.aif.dispersion.fixed_value = fixed_dispersion;
  options.bounds = tcm::default_bounds(
      options.model, options.aif);
  options.max_iterations = max_iterations;
  options.gradient_tolerance = gradient_tolerance;
  options.step_tolerance = step_tolerance;
  options.cost_tolerance = cost_tolerance;
  options.fd_step = fd_step;
  options.lambda_initial = lambda_initial;
  options.use_linear_warm_start = use_linear_warm_start;
  return options;
}

nb::ndarray<nb::numpy, double, nb::ndim<1>> wrap_vector(
    std::vector<double>& values,
    const nb::handle& owner) {
  return nb::ndarray<nb::numpy, double, nb::ndim<1>>(
      values.data(),
      {values.size()},
      owner);
}

nb::ndarray<nb::numpy, int32_t, nb::ndim<1>> wrap_vector(
    std::vector<int32_t>& values,
    const nb::handle& owner) {
  return nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(
      values.data(),
      {values.size()},
      owner);
}

nb::dict make_batch_result_dict(
    std::unique_ptr<tcm::TacBatchResult> result,
    const tcm::SolverOptions& options) {
  tcm::TacBatchResult* raw = result.release();
  nb::capsule owner(raw, [](void* pointer) noexcept {
    delete static_cast<tcm::TacBatchResult*>(pointer);
  });

  nb::dict out;
  out[nb::str("K1")] = wrap_vector(raw->K1, owner);
  out[nb::str("k2")] = wrap_vector(raw->k2, owner);
  out[nb::str("k3")] = wrap_vector(raw->k3, owner);
  out[nb::str("vB")] = wrap_vector(raw->vB, owner);
  out[nb::str("Ki")] = wrap_vector(raw->Ki, owner);
  out[nb::str("rmse")] = wrap_vector(raw->rmse, owner);
  out[nb::str("weighted_cost")] = wrap_vector(raw->weighted_cost, owner);
  out[nb::str("iterations")] = wrap_vector(raw->iterations, owner);
  out[nb::str("status")] = wrap_vector(raw->status, owner);
  if (tcm::is_reversible(options.model)) {
    out[nb::str("k4")] = wrap_vector(raw->k4, owner);
  }
  if (options.aif.delay.fitted || std::abs(options.aif.delay.fixed_value) > 1e-12) {
    out[nb::str("delay")] = wrap_vector(raw->delay, owner);
  }
  if (options.aif.dispersion.fitted || std::abs(options.aif.dispersion.fixed_value) > 1e-12) {
    out[nb::str("dispersion")] = wrap_vector(raw->dispersion, owner);
  }
  out[nb::str("fitted_tacs")] = nb::int_(raw->fitted_tacs);
  return out;
}

nb::dict fit_tacs_impl(
    const CpuArray& tacs,
    const CpuArray& time,
    const CpuArray& aif,
    nb::object weights,
    const std::string& model,
    bool fit_vb,
    double fixed_vb,
    bool fit_delay,
    double fixed_delay,
    bool fit_dispersion,
    double fixed_dispersion,
    int threads,
    int max_iterations,
    double gradient_tolerance,
    double step_tolerance,
    double cost_tolerance,
    double fd_step,
    double lambda_initial,
    bool use_linear_warm_start) {
  if (tacs.ndim() != 2) {
    throw std::runtime_error("tacs must have shape (T, N)");
  }

  const std::size_t t_count = tacs.shape(0);
  const std::size_t tac_count = tacs.shape(1);
  const tcm::SolverOptions options = make_solver_options(
      model,
      fit_vb,
      fixed_vb,
      fit_delay,
      fixed_delay,
      fit_dispersion,
      fixed_dispersion,
      max_iterations,
      gradient_tolerance,
      step_tolerance,
      cost_tolerance,
      fd_step,
      lambda_initial,
      use_linear_warm_start);

  const std::size_t time_columns = infer_columns(time, t_count, "time");
  const std::size_t aif_columns = infer_columns(aif, t_count, "aif");
  std::vector<double> weights_data;
  std::size_t weights_columns = 0;
  if (!weights.is_none()) {
    const CpuArray weights_array = nb::cast<CpuArray>(weights);
    weights_columns = infer_columns(weights_array, t_count, "weights");
    weights_data = copy_array(weights_array);
  }

  const tcm::TacBatchData batch = tcm::make_tac_batch_data(
      copy_array(time),
      time_columns,
      copy_array(tacs),
      t_count,
      tac_count,
      copy_array(aif),
      aif_columns,
      std::move(weights_data),
      weights_columns);

  std::unique_ptr<tcm::TacBatchResult> result;
  {
    nb::gil_scoped_release release;
    result = std::make_unique<tcm::TacBatchResult>(
        tcm::fit_tac_batch(batch, options, threads));
  }
  return make_batch_result_dict(std::move(result), options);
}

nb::dict fit_one_tac_impl(
    const CpuVector& tac,
    const CpuVector& time,
    const CpuVector& aif,
    nb::object weights,
    const std::string& model,
    bool fit_vb,
    double fixed_vb,
    bool fit_delay,
    double fixed_delay,
    bool fit_dispersion,
    double fixed_dispersion,
    int max_iterations,
    double gradient_tolerance,
    double step_tolerance,
    double cost_tolerance,
    double fd_step,
    double lambda_initial,
    bool use_linear_warm_start) {
  if (tac.shape(0) != time.shape(0) || aif.shape(0) != time.shape(0)) {
    throw std::runtime_error("tac, time, and aif must have the same length");
  }

  std::vector<double> weights_data;
  if (!weights.is_none()) {
    const CpuVector weights_vector = nb::cast<CpuVector>(weights);
    if (weights_vector.shape(0) != time.shape(0)) {
      throw std::runtime_error("weights length must match the time vector");
    }
    weights_data = copy_array(weights_vector);
  }

  const tcm::SolverOptions options = make_solver_options(
      model,
      fit_vb,
      fixed_vb,
      fit_delay,
      fixed_delay,
      fit_dispersion,
      fixed_dispersion,
      max_iterations,
      gradient_tolerance,
      step_tolerance,
      cost_tolerance,
      fd_step,
      lambda_initial,
      use_linear_warm_start);

  const tcm::TacBatchData batch = tcm::make_tac_batch_data(
      copy_array(time),
      1,
      copy_array(tac),
      time.shape(0),
      1,
      copy_array(aif),
      1,
      std::move(weights_data),
      weights.is_none() ? 0 : 1);

  tcm::TacBatchResult batch_result;
  {
    nb::gil_scoped_release release;
    batch_result = tcm::fit_tac_batch(batch, options, 0);
  }

  nb::dict out;
  out[nb::str("K1")] = nb::float_(batch_result.K1[0]);
  out[nb::str("k2")] = nb::float_(batch_result.k2[0]);
  out[nb::str("k3")] = nb::float_(batch_result.k3[0]);
  out[nb::str("vB")] = nb::float_(batch_result.vB[0]);
  out[nb::str("Ki")] = nb::float_(batch_result.Ki[0]);
  out[nb::str("rmse")] = nb::float_(batch_result.rmse[0]);
  out[nb::str("weighted_cost")] = nb::float_(batch_result.weighted_cost[0]);
  out[nb::str("iterations")] = nb::int_(batch_result.iterations[0]);
  out[nb::str("status")] = nb::int_(batch_result.status[0]);
  if (tcm::is_reversible(options.model)) {
    out[nb::str("k4")] = nb::float_(batch_result.k4[0]);
  }
  if (options.aif.delay.fitted || std::abs(options.aif.delay.fixed_value) > 1e-12) {
    out[nb::str("delay")] = nb::float_(batch_result.delay[0]);
  }
  if (options.aif.dispersion.fitted || std::abs(options.aif.dispersion.fixed_value) > 1e-12) {
    out[nb::str("dispersion")] = nb::float_(batch_result.dispersion[0]);
  }
  return out;
}

nb::ndarray<nb::numpy, double, nb::ndim<1>> evaluate_model_impl(
    const CpuVector& time,
    const CpuVector& aif,
    const std::string& model,
    double K1,
    double k2,
    double k3,
    double k4,
    double vB,
    double delay,
    double dispersion) {
  if (time.shape(0) != aif.shape(0)) {
    throw std::runtime_error("time and aif must have the same length");
  }

  tcm::ParameterVector params;
  params.K1 = K1;
  params.k2 = k2;
  params.k3 = k3;
  params.k4 = k4;
  params.vB = vB;
  params.delay = delay;
  params.dispersion = dispersion;

  std::vector<double> predicted;
  const bool ok = tcm::evaluate_model(
      tcm::make_model(model),
      params,
      copy_time_minutes(time),
      copy_array(aif),
      predicted);
  if (!ok) {
    throw std::runtime_error("evaluate_model failed for the provided parameters");
  }

  auto values = std::make_unique<std::vector<double>>(std::move(predicted));
  auto* raw = values.release();
  nb::capsule owner(raw, [](void* pointer) noexcept {
    delete static_cast<std::vector<double>*>(pointer);
  });
  return wrap_vector(*raw, owner);
}

nb::dict default_bounds_impl(
    const std::string& model,
    bool fit_vb,
    double fixed_vb,
    bool fit_delay,
    double fixed_delay,
    bool fit_dispersion,
    double fixed_dispersion) {
  const tcm::Model kinetics = tcm::make_model(model, tcm::blood_fraction_setting(fit_vb, fixed_vb));
  const tcm::AifOptions aif_options{{fit_delay, fixed_delay}, {fit_dispersion, fixed_dispersion}};
  const tcm::SolverBounds bounds = tcm::default_bounds(kinetics, aif_options);
  const int count = tcm::active_parameter_count(kinetics, aif_options);

  nb::dict out;
  for (int index = 0; index < count; ++index) {
    const std::string name = tcm::parameter_name(kinetics, aif_options, index);
    out[nb::str(name.c_str())] =
        nb::make_tuple(bounds.lower[static_cast<std::size_t>(index)], bounds.upper[static_cast<std::size_t>(index)]);
  }
  return out;
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "Nanobind bindings for TAC-based dynamic PET compartment-model fitting.";

  nb::enum_<tcm::FitStatus>(m, "FitStatus")
      .value("Failed", tcm::FitStatus::Failed)
      .value("GradientConverged", tcm::FitStatus::GradientConverged)
      .value("StepConverged", tcm::FitStatus::StepConverged)
      .value("CostConverged", tcm::FitStatus::CostConverged)
      .value("MaxIterations", tcm::FitStatus::MaxIterations)
      .value("InvalidInput", tcm::FitStatus::InvalidInput);

  m.def(
      "fit_tacs",
      &fit_tacs_impl,
      "tacs"_a,
      "time"_a,
      "aif"_a,
      "weights"_a = nb::none(),
      "model"_a = "rev",
      "fit_vb"_a = true,
      "fixed_vb"_a = 0.05,
      "fit_delay"_a = false,
      "fixed_delay"_a = 0.0,
      "fit_dispersion"_a = false,
      "fixed_dispersion"_a = 0.0,
      "threads"_a = 0,
      "max_iterations"_a = 40,
      "gradient_tolerance"_a = 1e-6,
      "step_tolerance"_a = 1e-6,
      "cost_tolerance"_a = 1e-9,
      "fd_step"_a = 1e-4,
      "lambda_initial"_a = 1e-2,
      "use_linear_warm_start"_a = true);

  m.def(
      "fit_one_tac",
      &fit_one_tac_impl,
      "tac"_a,
      "time"_a,
      "aif"_a,
      "weights"_a = nb::none(),
      "model"_a = "rev",
      "fit_vb"_a = true,
      "fixed_vb"_a = 0.05,
      "fit_delay"_a = false,
      "fixed_delay"_a = 0.0,
      "fit_dispersion"_a = false,
      "fixed_dispersion"_a = 0.0,
      "max_iterations"_a = 40,
      "gradient_tolerance"_a = 1e-6,
      "step_tolerance"_a = 1e-6,
      "cost_tolerance"_a = 1e-9,
      "fd_step"_a = 1e-4,
      "lambda_initial"_a = 1e-2,
      "use_linear_warm_start"_a = true);

  m.def(
      "evaluate_model",
      &evaluate_model_impl,
      "time"_a,
      "aif"_a,
      "model"_a,
      "K1"_a,
      "k2"_a,
      "k3"_a,
      "k4"_a = 0.0,
      "vB"_a = 0.05,
      "delay"_a = 0.0,
      "dispersion"_a = 0.0);

  m.def(
      "default_bounds",
      &default_bounds_impl,
      "model"_a = "rev",
      "fit_vb"_a = true,
      "fixed_vb"_a = 0.05,
      "fit_delay"_a = false,
      "fixed_delay"_a = 0.0,
      "fit_dispersion"_a = false,
      "fixed_dispersion"_a = 0.0);

  m.def("default_frame_weights", [](const CpuVector& time) {
    const std::vector<double> weights = tcm::default_frame_weights(copy_time_minutes(time));
    auto heap_weights = std::make_unique<std::vector<double>>(weights);
    auto* raw = heap_weights.release();
    nb::capsule owner(raw, [](void* pointer) noexcept {
      delete static_cast<std::vector<double>*>(pointer);
    });
    return wrap_vector(*raw, owner);
  }, "time"_a);
}
