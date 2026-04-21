"""
Scatter version of torch.distributions.Distribution
Skip validity check of input arguments to reduce cpu-gpu synchronization overhead.
"""

from typing import cast

import torch
import torch.nn.functional as F
from torch.distributions import Distribution
from torch.distributions.utils import broadcast_all, logits_to_probs, probs_to_logits

from DMGG.scatter.operations import (
    scatter_argmax,
    scatter_logsumexp,
    scatter_softmax,
    scatter_sum,
)


class ScatterCategorical(Distribution):
    """
    Categorical distribution for graph-batched data using scatter operations.
    Receives pre-computed probabilities (or logits) and performs scatter-based sampling/entropy.
    """

    _probs: torch.Tensor | None = None
    _logits: torch.Tensor | None = None

    def __init__(
        self,
        probs: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        batch: torch.LongTensor | None = None,
        ptr: torch.LongTensor | None = None,
    ):
        """
        Args:
            probs: [G, ] = [G1 + G2 + ... + GB], probabilities of each element
            logits: [G, ] = [G1 + G2 + ... + GB], unnormalized log probabilities of each element
            batch: [G, ] = [0, 0, ..., 0, 1, 1, ..., 1, ..., B-1, B-1, ..., B-1], group index for each element
            ptr: [B+1, ] = [0, G1, G1+G2, ..., G1+G2+...+GB], pointer for each group
        """
        device = logits.device if logits is not None else probs.device  # type: ignore[attr-defined]
        assert (logits is None) != (
            probs is None
        ), "Either logits or probs must be provided, but not both"

        if batch is None:
            assert ptr is not None, "Either batch or ptr must be provided"
            num_batch = len(ptr) - 1
            num_samples = ptr[1:] - ptr[:-1]

            batch = cast(
                torch.LongTensor,
                torch.arange(
                    num_batch, dtype=torch.int64, device=device
                ).repeat_interleave(num_samples),
            )
        if ptr is None:
            assert batch is not None, "Either batch or ptr must be provided"
            num_batch = int(batch.max()) + 1
            num_samples = torch.bincount(batch, minlength=num_batch)
            ptr = cast(
                torch.LongTensor,
                torch.zeros(num_batch + 1, device=device, dtype=torch.int64),
            )
            ptr[1:] = num_samples.cumsum(0)

        self.batch, self.ptr = batch, ptr
        self.num_batch = len(ptr) - 1

        if logits is not None:
            logits = logits.to(torch.float32)

            # Normalize
            norm = scatter_logsumexp(logits, self.batch, dim=0, dim_size=self.num_batch)
            self._logits = logits - norm[self.batch]
            self._probs = None
        else:
            probs = probs.to(torch.float32)  # type: ignore[attr-defined]

            # Normalize
            norm = scatter_sum(probs, self.batch, dim=0, dim_size=self.num_batch)
            self._probs = probs / norm[self.batch]
            self._logits = None

    def sample(self) -> torch.LongTensor:  # type: ignore[override]
        """
        Sample from the distribution using Gumbel-Max trick

        sampling from categorical distribution
        <=> add Gumbel noise to standard logits(log_prob) and take argmax

        Return:
            indices: [B, ] (Global indices)
        """
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.logits)))
        score = self.logits + gumbel_noise

        return scatter_argmax(score, self.batch, dim=0, dim_size=self.num_batch)

    def log_prob(self, value: torch.LongTensor) -> torch.Tensor:  # type: ignore[override]
        """
        Arg:
            value: [B, ], Global indices
        Return:
            log_prob: [B, ]
        """
        return self.logits[value.to(torch.int64)]

    def entropy(self) -> torch.Tensor:
        """
        Compute entropy per batch: H = - sum(p * log(p))
        Return:
            entropy: [B, ]
        """
        logits = torch.clamp(self.logits, min=torch.finfo(self.logits.dtype).min)
        p_log_p = logits * self.probs
        return -scatter_sum(p_log_p, self.batch, dim=0, dim_size=self.num_batch)

    @property
    def logits(self) -> torch.Tensor:
        if self._logits is None:
            # self._logits = probs_to_logits(self.probs)
            self._logits = torch.where(
                self.probs > 0, probs_to_logits(self.probs), -torch.inf
            )
        return self._logits

    @property
    def probs(self) -> torch.Tensor:
        if self._probs is None:
            self._probs = scatter_softmax(
                self.logits, self.batch, dim=0, dim_size=self.num_batch
            )

        return self._probs


class ScatterBernoulli(Distribution):
    _probs: torch.Tensor | None
    _logits: torch.Tensor | None

    def __init__(
        self, probs: torch.Tensor | None = None, logits: torch.Tensor | None = None
    ):
        assert (probs is None) != (
            logits is None
        ), "Either `probs` or `logits` must be specified, but not both."

        if probs is not None:
            (self._probs,) = broadcast_all(probs)
            self._probs = cast(torch.Tensor, self._probs)
            self._logits = None
            self._batch_shape = self._probs.size()

        else:
            (self._logits,) = broadcast_all(logits)  # type:ignore[arg-type]
            self._logits = cast(torch.Tensor, self._logits)
            self._probs = None
            self._batch_shape = self._logits.size()

    @property
    def mean(self) -> torch.Tensor:
        return self.probs

    @property
    def mode(self) -> torch.Tensor:
        mode = (self.probs >= 0.5).to(self.probs)
        mode[self.probs == 0.5] = torch.nan
        return mode

    @property
    def variance(self) -> torch.Tensor:
        return self.probs * (1 - self.probs)

    @property
    def logits(self) -> torch.Tensor:
        if self._logits is None:
            self._logits = probs_to_logits(self.probs, is_binary=True)
        return self._logits

    @property
    def probs(self) -> torch.Tensor:
        if self._probs is None:
            self._probs = logits_to_probs(self.logits, is_binary=True)
        return self._probs

    def sample(self, sample_shape: tuple[int, ...] = tuple()) -> torch.Tensor:  # type: ignore[override]
        shape = sample_shape + self._batch_shape

        with torch.no_grad():
            return torch.bernoulli(self.probs.expand(shape))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        logits, value = broadcast_all(self.logits, value)

        return -F.binary_cross_entropy_with_logits(logits, value, reduction="none")

    def entropy(self) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            self.logits, self.probs, reduction="none"
        )
