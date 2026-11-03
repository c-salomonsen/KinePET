#include "lls.hpp"
#include "tcm_model.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
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
        const auto model = reversible ? tcm::ModelKind::Reversible2TCM : tcm::ModelKind::Irreversible2TCM;
        const bool fit_vb = blood_mode == 0;
        const double fixed_vb = blood_mode == 1 ? 0.0 : 0.05;
        std::vector<double> time = {0., .0833, .1667, .25, .3333, .5, .75, 1., 1.5, 2., 3., 5., 8., 12., 20., 30., 45.};
        if (positive_first_time) time[0] = .02;
        std::vector<double> aif;
        for (double t : time) aif.push_back(12. * std::exp(-t / .3) + 1.5 * std::exp(-t / 6.));
        tcm::ParameterVector truth{.09, .22, .11, reversible ? .06 : 0., fit_vb ? .04 : fixed_vb};
        std::vector<double> tac;
        require(tcm::evaluate_model(model, truth, time, aif, tac), "forward evaluation failed");
        const auto bounds = tcm::default_bounds(model, fit_vb, fixed_vb, false, 0., false, 0.);
        const auto warm = tcm::linear_warm_start(model, fit_vb, fixed_vb, false, 0., false, 0., bounds, time, aif, tac);
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
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
