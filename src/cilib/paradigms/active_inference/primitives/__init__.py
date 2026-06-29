"""Vendored pure math kernels for the active-inference paradigm."""
from .gaussian_info import (
    info_mean, info_cov, log_evidence, add_information,
    fisher_deposit, fisher_deposit_weighted,
    predictive_logpdf, predictive_logpdf_perchannel, savage_dickey, schur_marginalize,
    spike_prior, zero_edge_prior, row_stochastic,
    softmax, entropy,
)
