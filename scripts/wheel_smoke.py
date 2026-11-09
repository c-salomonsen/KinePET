"""Run outside the checkout against an installed wheel; NumPy is sufficient."""
import numpy as np
import kinepet


def main():
    time = np.array([0., .0833, .1667, .25, .3333, .5, .75, 1., 1.5, 2., 3., 5., 8., 12., 20., 30., 45.])
    aif = 12 * np.exp(-time / .3) + 1.5 * np.exp(-time / 6)
    for kind in (kinepet.Irreversible2TCM, kinepet.Reversible2TCM):
        for vb in (kinepet.FittedVB(), kinepet.FixedVB(.04)):
            model = kind(vb=vb)
            k4 = .06 if kind is kinepet.Reversible2TCM else 0.
            tac = kinepet.evaluate_model(time, aif, model=model, K1=.09, k2=.22, k3=.11, k4=k4, vB=.04)
            fit = kinepet.fit_one_tac(tac, time, aif, model=model, max_iterations=200,
                                      gradient_tolerance=1e-10, cost_tolerance=1e-16)
            assert fit.status_code in (1, 2, 3), fit
            assert np.isfinite(fit.weighted_cost) and fit.rmse < 1e-5, fit
            np.testing.assert_allclose([fit.K1, fit.k2, fit.k3, fit.vB], [.09, .22, .11, .04], rtol=.01, atol=1e-4)
            if k4:
                np.testing.assert_allclose(fit.k4, k4, rtol=.01, atol=1e-4)
    print("Installed-wheel fits passed for both models and blood-fraction settings.")


if __name__ == "__main__":
    main()
