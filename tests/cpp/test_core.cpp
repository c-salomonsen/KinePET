#include "lls.hpp"
#include "lls/formulations.hpp"
#include "tcm_model.hpp"
#include "aif.hpp"
#include "solver.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void check_close(double actual, double expected, const std::string& name) {
  require(std::isfinite(actual) && std::abs(actual - expected) <=
              1e-11 + 1e-9 * std::abs(expected), name + " differs from the pre-refactor baseline");
}

// Readable, text-based baselines preserve LLS initialization independently of
// nonlinear convergence. They were recorded before the readability refactor.
void check_warm_starts(std::istream& expected, bool record) {
  for (bool reversible : {false, true}) {
    for (int blood_mode : {0, 1, 2}) {  // fitted, fixed zero, fixed nonzero
      for (bool positive_first_time : {false, true}) {
        const bool fit_vb = blood_mode == 0;
        const double fixed_vb = blood_mode == 1 ? 0.0 : 0.05;
        const auto model = tcm::make_model(reversible ? "rev" : "irr", tcm::blood_fraction_setting(fit_vb, fixed_vb));
        std::vector<double> time = {0., .0833, .1667, .25, .3333, .5, .75, 1., 1.5, 2., 3., 5., 8., 12., 20., 30., 45.};
        if (positive_first_time) time[0] = .02;
        std::vector<double> aif;
        for (double t : time) aif.push_back(12. * std::exp(-t / .3) + 1.5 * std::exp(-t / 6.));
        tcm::ParameterVector truth{.09, .22, .11, reversible ? .06 : 0., fit_vb ? .04 : fixed_vb};
        std::vector<double> tac;
        require(tcm::evaluate_model(model, truth, time, aif, tac), "forward evaluation failed");
        const auto bounds = tcm::default_bounds(model, {});
        const auto warm = tcm::warm_start(model, {}, bounds, time, aif, tac);
        const std::string label = std::string(reversible ? "rev" : "irr") + "_" +
            std::to_string(blood_mode) + "_" + (positive_first_time ? "positive" : "zero");
        const std::vector<double> values = {warm.valid ? 1. : 0., warm.params.K1, warm.params.k2,
            warm.params.k3, warm.params.k4, warm.params.vB, warm.params.delay, warm.params.dispersion};
        if (record) {
          std::cout << label;
          for (double value : values) std::cout << ' ' << std::setprecision(17) << value;
          std::cout << '\n';
        } else {
          std::string expected_label;
          require(static_cast<bool>(expected >> expected_label), "missing baseline " + label);
          require(label == expected_label, "baseline scenario order differs");
          for (double value : values) {
            double reference = 0.;
            require(static_cast<bool>(expected >> reference), "missing value in " + label);
            check_close(value, reference, label);
          }
        }
      }
    }
  }
  if (!record) {
    std::string extra;
    require(!(expected >> extra), "unexpected trailing baseline data");
  }
}

void check_reversible_recovery() {
  // Construct the derivation's exact regression coefficients, bypassing
  // quadrature and ridge so mistakes cannot be hidden by the nonlinear fit.
  for (double k4 : {0.0, 0.06, 0.3}) {
    for (double vb : {0.0, 0.04, 0.7}) {
      const double K1 = .09, k2 = .22, k3 = .11;
      const double alpha = k2 + k3 + k4;
      const double beta = k2 * k4;
      const tcm::lls::Coefficients coefficients{
          -alpha, -beta, vb, vb * alpha + (1. - vb) * K1,
          vb * beta + (1. - vb) * K1 * (k3 + k4)};
      tcm::ParameterVector recovered;
      require(tcm::lls::recover_reversible_fitted_vb(coefficients, recovered), "exact recovery rejected");
      check_close(recovered.K1, K1, "recovered K1");
      check_close(recovered.k2, k2, "recovered k2");
      check_close(recovered.k3, k3, "recovered k3");
      check_close(recovered.k4, k4, "recovered k4");
      check_close(recovered.vB, vb, "recovered vB");
    }
  }
  tcm::ParameterVector params;
  require(!tcm::lls::recover_reversible_fitted_vb({-.39, -.0132, 1., .39, .0132}, params),
          "pure vascular signal has no identifiable tissue parameters");
  require(!tcm::lls::recover_reversible_fitted_vb({-.39, -.0132, 0., 0., 0.}, params),
          "zero K1 must reject recovery");
  require(!tcm::lls::recover_reversible_fitted_vb({-.17, 0., 0., .09, .09 * .17}, params),
          "zero k2 must reject recovery");
}

void check_parameter_layouts() {
  const tcm::ParameterVector truth{.12, .23, .34, .045, .056, -.03, .025};
  for (const std::string name : {"irr", "rev"}) {
    for (bool fitted_vb : {false, true}) {
      for (bool fitted_delay : {false, true}) {
        for (bool fitted_dispersion : {false, true}) {
          const auto model = tcm::make_model(name, tcm::blood_fraction_setting(fitted_vb, truth.vB));
          const tcm::AifOptions aif_options{{fitted_delay, truth.delay}, {fitted_dispersion, truth.dispersion}};
          std::vector<std::string> names{"K1", "k2", "k3"};
          if (name == "rev") names.push_back("k4");
          if (fitted_vb) names.push_back("vB");
          if (fitted_delay) names.push_back("delay");
          if (fitted_dispersion) names.push_back("dispersion");
          require(tcm::active_parameter_count(model, aif_options) == static_cast<int>(names.size()),
                  "wrong active parameter count");
          std::array<double, tcm::kMaxParameterCount> packed{};
          tcm::pack_active(truth, model, aif_options, packed);
          const auto recovered = tcm::unpack_active(packed, model, aif_options);
          check_close(recovered.K1, truth.K1, "packed K1");
          check_close(recovered.k2, truth.k2, "packed k2");
          check_close(recovered.k3, truth.k3, "packed k3");
          check_close(recovered.k4, name == "rev" ? truth.k4 : 0., "packed k4");
          check_close(recovered.vB, truth.vB, "packed vB");
          check_close(recovered.delay, truth.delay, "packed delay");
          check_close(recovered.dispersion, truth.dispersion, "packed dispersion");
          for (std::size_t i = 0; i < names.size(); ++i) {
            require(tcm::parameter_name(model, aif_options, static_cast<int>(i)) == names[i],
                    "wrong exported parameter order");
          }
        }
      }
    }
  }
}

void check_warm_start_rejection() {
  const std::vector<double> time{0., .1, .2, .5, 1., 2.};
  const std::vector<double> aif{0., 10., 5., 3., 1., .5};
  const std::vector<double> zero(time.size(), 0.);
  for (const std::string name : {"irr", "rev"}) {
    for (bool fitted : {false, true}) {
      const auto model = tcm::make_model(name, tcm::blood_fraction_setting(fitted, 0.));
      const auto bounds = tcm::default_bounds(model, {});
      require(!tcm::warm_start(model, {}, bounds, time, aif, zero).valid,
              "zero signal should reject the LLS warm start");
      require(!tcm::warm_start(model, {}, bounds, time, {}, zero).valid,
              "mismatched input must reject the LLS warm start");
    }
  }
}

void check_invalid_solver_inputs() {
  const std::vector<double> time{0., .1, .2, .5, 1., 2.};
  const std::vector<double> aif{0., 10., 5., 3., 1., .5};
  const std::vector<double> weights(time.size(), 1.);
  tcm::SolverOptions options;
  options.bounds = tcm::default_bounds(options.model, {});
  options.max_iterations = 0;
  require(tcm::fit_tac(options, time, aif, weights, aif).status == tcm::FitStatus::MaxIterations,
          "zero iterations must retain initialization-only mode");
  options.bounds.upper[0] = options.bounds.lower[0];
  require(tcm::fit_tac(options, time, aif, weights, aif).status == tcm::FitStatus::InvalidInput,
          "equal bounds must be rejected before division by span");
  options.bounds = tcm::default_bounds(options.model, {});
  auto initial = tcm::default_initial_guess(options.model, {});
  initial.K1 = std::numeric_limits<double>::quiet_NaN();
  require(tcm::fit_tac_with_initial_guess(options, time, aif, weights, aif, initial).status ==
              tcm::FitStatus::InvalidInput, "nonfinite explicit start must be rejected");
  for (double delay : {-.03, .03}) {
    for (double dispersion : {0., .02}) {
      std::vector<double> expected;
      tcm::transform_aif(time, aif, delay, dispersion, expected);
      auto in_place = aif;
      tcm::transform_aif(time, in_place, delay, dispersion, in_place);
      for (std::size_t i = 0; i < time.size(); ++i) {
        check_close(in_place[i], expected[i], "aliased AIF transform");
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: test_core FIXTURE | --record");
    const bool record = std::string(argv[1]) == "--record";
    std::ifstream fixture;
    if (!record) {
      fixture.open(argv[1]);
      require(fixture.is_open(), "cannot open warm-start fixture");
    }
    check_warm_starts(fixture, record);
    check_reversible_recovery();
    check_parameter_layouts();
    check_warm_start_rejection();
    check_invalid_solver_inputs();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
