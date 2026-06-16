"""Negative sampling for constraint-aware contrastive learning.

Preferred: ChannelAlignedSampler (real value perturbations, contribution C7).
Legacy:    CHAPNegativeSampler (stub CHAP-E, kept for backward compat).
"""

from gsr_cacl.negative_sampler.chap import (
    CHAPNegativeSampler,
    PerturbedTable,
    apply_chap_a,
    apply_chap_s,
    apply_chap_e,
)
from gsr_cacl.negative_sampler.channel_aligned import (
    ChannelAlignedSampler,
    ChannelNegative,
    make_negative,
)

__all__ = [
    # preferred
    "ChannelAlignedSampler",
    "ChannelNegative",
    "make_negative",
    # legacy
    "CHAPNegativeSampler",
    "PerturbedTable",
    "apply_chap_a",
    "apply_chap_s",
    "apply_chap_e",
]
