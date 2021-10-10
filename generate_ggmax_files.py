#!/usr/bin/env python3
# coding: utf-8

""" This script is an example of how the automatic sampler can be used """

from pathlib import Path

import numpy as np

from ggmax_sampler import get_best_points_for_ggmax_sampling

DATADIR = Path(__file__).parent / "data"
DATADIR.mkdir(exist_ok=True)


class GGmaxHyperbolic:
    def __init__(self, reference_shear_strain: float):
        self.reference_shear_strain = reference_shear_strain

    def __call__(self, e: float) -> float:
        return 1 / (1 + e / self.reference_shear_strain)


def main():
    gammas = [5.04e-4, 3.96e-4, 3.23e-4, 1.49e-3, 5.04e-4]
    xmin = 1e-7
    xmax = 1e0
    N = 50
    for i, g in enumerate(gammas):
        ggmax = GGmaxHyperbolic(g)
        auto_e = get_best_points_for_ggmax_sampling(ggmax, xmin, xmax, N)
        log_e = np.logspace(np.log10(xmin), np.log10(xmax), N)

        np.savetxt(
            DATADIR / f"ggmax_layer_{N}_auto_{i}.txt",
            np.vstack((auto_e, ggmax(auto_e))).T,
        )
        np.savetxt(
            DATADIR / f"ggmax_layer_{N}_log_{i}.txt", np.vstack((log_e, ggmax(log_e))).T
        )


if __name__ == "__main__":
    main()
