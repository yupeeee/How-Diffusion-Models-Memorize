"""CUDA binary64 interval arithmetic for fixed-cache refinement.

All numerical arrays and arithmetic stay on the selected NVIDIA CUDA device.
Only scalar sign decisions and final two-endpoint serialization reach the host.
This is binary64 enclosure arithmetic, not arbitrary-decimal precision.

Each eager CUDA add/multiply/divide/sqrt is rounded to nearest; one nextafter
outward encloses that operation, including gradual underflow and overflow.
CUDA documents this contract (fast-math flags do not change double precision):
https://docs.nvidia.com/cuda/archive/12.1.1/floating-point/index.html#cuda-and-floating-point
PyTorch's CUDA sqrt and frexp dispatch to their CUDA/C++ counterparts:
https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/UnaryOpsKernel.cu

We deliberately do NOT assume that CUDA exp/log are correctly rounded. exp
uses an enclosed Taylor polynomial after ln(2) reduction; log uses the atanh
series after exact frexp decomposition. Both include analytic remainder bounds.
Reductions use explicit pairwise outward additions, never an unchecked sum,
BLAS dot product, or tensor-core operation. Keep this backend in eager mode:
fusing an operation with its following outward rounding changes the contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


GPU_INTERVAL_VERSION = "cuda-fp64-outward-series-1"
# Adjacent binary64 bounds on ln(2). The exact rational identity
# ln(2) = 2*sum_{k>=0} (1/3)**(2*k+1)/(2*k+1), with its geometric tail,
# proves these constants; a source regression specifies that proof with Fraction.
_LN2_LOWER = float.fromhex("0x1.62e42fefa39efp-1")
_LN2_UPPER = float.fromhex("0x1.62e42fefa39f0p-1")
_SMALLEST = float.fromhex("0x0.0000000000001p-1022")
_LARGEST = float.fromhex("0x1.fffffffffffffp+1023")


class ArithmeticBudgetExceeded(RuntimeError):
    """The deterministic per-row GPU interval-work budget was exhausted."""


def _device(value):
    device = torch.device(value)
    if device.type != "cuda" or getattr(torch.version, "hip", None):
        raise ValueError("Numerical interval refinement requires NVIDIA CUDA; CPU fallback is disabled")
    return device


def _down(value):
    return torch.nextafter(value, torch.full_like(value, -torch.inf))


def _up(value):
    return torch.nextafter(value, torch.full_like(value, torch.inf))


@dataclass(frozen=True)
class GpuInterval:
    lower: torch.Tensor
    upper: torch.Tensor

    def __post_init__(self):
        if not isinstance(self.lower, torch.Tensor) or not isinstance(self.upper, torch.Tensor):
            raise TypeError("GPU interval endpoints must be tensors")
        _device(self.lower.device)
        if (self.lower.dtype != torch.float64 or self.upper.dtype != torch.float64
                or self.lower.device != self.upper.device or self.lower.shape != self.upper.shape):
            raise ValueError("GPU interval endpoints must be equally shaped CUDA float64 tensors")

    @classmethod
    def point(cls, value, *, device=None):
        if device is None:
            device = value.device if isinstance(value, torch.Tensor) else "cuda"
        device = _device(device)
        value = torch.as_tensor(value, device=device, dtype=torch.float64)
        # A nonfinite source input is never a finite-input sign certificate.
        finite = torch.isfinite(value)
        return cls(torch.where(finite, value, -torch.inf), torch.where(finite, value, torch.inf))

    def __getitem__(self, index):
        return GpuInterval(self.lower[index], self.upper[index])

    def unsqueeze(self, dim):
        return GpuInterval(self.lower.unsqueeze(dim), self.upper.unsqueeze(dim))

    def reshape(self, *shape):
        return GpuInterval(self.lower.reshape(*shape), self.upper.reshape(*shape))

    @property
    def sign(self):
        if self.lower.numel() != 1:
            raise ValueError("Only a scalar interval has one sign")
        lo, hi = self.floats()
        if math.isnan(lo) or math.isnan(hi) or lo > hi:
            return "unresolved"
        if lo > 0 and math.isfinite(lo):
            return "positive"
        if hi < 0 and math.isfinite(hi):
            return "negative"
        if lo == hi == 0:
            return "zero"
        return "unresolved"

    def floats(self):
        """Serialize two already-outward binary64 endpoints, never raw vectors."""
        if self.lower.numel() != 1:
            raise ValueError("Only scalar intervals can be serialized as two floats")
        return float(self.lower.item()), float(self.upper.item())


class GpuDirected:
    """Vectorized eager CUDA intervals with deterministic scalar-lane accounting.

    Charge one lane per point input/add/sub, two per bounded input lane,
    four per multiply/divide, and two per
    square/sqrt. Transcendentals charge their composed primitive operations;
    pairwise reductions charge every addition lane. Integer indexing, endpoint
    comparisons, and nextafter belong to the primitive and are not extra lanes.
    A context belongs to one row; broadcasting charges the full output shape.
    """

    def __init__(self, max_products=2_000_000, *, device="cuda"):
        if not isinstance(max_products, int) or isinstance(max_products, bool) or max_products < 0:
            raise ValueError("GPU interval budget must be a nonnegative integer")
        self.device = _device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.max_products = max_products
        self.products = 0
        self.evaluated_nodes = 0
        self.precision = 53  # binary significand bits, never decimal digits
        self._constants = {}

    def charge(self, count=1):
        if not isinstance(count, int) or count < 0:
            raise ValueError("GPU interval work charge must be a nonnegative integer")
        if self.products + count > self.max_products:
            raise ArithmeticBudgetExceeded("fixed per-row GPU interval operation budget exhausted")
        self.products += count

    def interval(self, value, upper=None):
        if isinstance(value, GpuInterval):
            if value.lower.device != self.device:
                raise ValueError("Interval and arithmetic context use different CUDA devices")
            return value
        lower = torch.as_tensor(value, dtype=torch.float64, device=self.device)
        if upper is None:
            self.charge(lower.numel())
            return GpuInterval.point(lower)
        upper = torch.as_tensor(upper, dtype=torch.float64, device=self.device)
        lower, upper = torch.broadcast_tensors(lower, upper)
        # Both supplied endpoint arrays are numerical inputs. Charge their
        # broadcast lanes, including a scalar endpoint expanded across a row.
        self.charge(2 * lower.numel())
        return self._finish(lower, upper)

    point = interval

    def _constant(self, value):
        # Exact small integer and binary64 constants are metadata; no host array
        # is built and reused constants do not consume repeated input charges.
        if value not in self._constants:
            self._constants[value] = GpuInterval.point(value, device=self.device)
        return self._constants[value]

    @staticmethod
    def _finish(lower, upper):
        invalid = torch.isnan(lower) | torch.isnan(upper) | (lower > upper)
        return GpuInterval(torch.where(invalid, -torch.inf, lower),
                           torch.where(invalid, torch.inf, upper))

    def _pair(self, x, y):
        x, y = self.interval(x), self.interval(y)
        shape = torch.broadcast_shapes(x.lower.shape, y.lower.shape)
        return x, y, math.prod(shape)

    def add(self, x, y):
        x, y, size = self._pair(x, y)
        self.charge(size)
        lower, upper = _down(x.lower + y.lower), _up(x.upper + y.upper)
        # Addition of an exact zero changes no finite binary64 endpoint.
        xzero, yzero = (x.lower == 0) & (x.upper == 0), (y.lower == 0) & (y.upper == 0)
        lower = torch.where(xzero, y.lower, torch.where(yzero, x.lower, lower))
        upper = torch.where(xzero, y.upper, torch.where(yzero, x.upper, upper))
        return self._finish(lower, upper)

    def neg(self, x):
        x = self.interval(x)
        return GpuInterval(-x.upper, -x.lower)

    def sub(self, x, y):
        x, y, size = self._pair(x, y)
        self.charge(size)
        lower, upper = _down(x.lower - y.upper), _up(x.upper - y.lower)
        # Subtracting equal exact point values is exactly zero, including the
        # same target atom and structural zero branch gaps.
        equal = (x.lower == x.upper) & (y.lower == y.upper) & (x.lower == y.lower) & torch.isfinite(x.lower)
        return self._finish(torch.where(equal, 0., lower), torch.where(equal, 0., upper))

    def mul(self, x, y):
        x, y, size = self._pair(x, y)
        self.charge(4 * size)
        corners = []
        for a in (x.lower, x.upper):
            for b in (y.lower, y.upper):
                product = a * b
                # Endpoint convention 0*(+/-infinity)=0 encloses multiplication
                # of any finite element in the extended input intervals.
                corners.append(torch.where((a == 0) | (b == 0), 0., product))
        values = torch.stack(torch.broadcast_tensors(*corners))
        lower, upper = _down(values.amin(0)), _up(values.amax(0))
        zero = ((x.lower == 0) & (x.upper == 0)) | ((y.lower == 0) & (y.upper == 0))
        return self._finish(torch.where(zero, 0., lower), torch.where(zero, 0., upper))

    def div(self, x, y):
        x, y, size = self._pair(x, y)
        self.charge(4 * size)
        values = torch.stack(torch.broadcast_tensors(*(a / b for a in (x.lower, x.upper) for b in (y.lower, y.upper))))
        lower, upper = _down(values.amin(0)), _up(values.amax(0))
        invalid = (y.lower <= 0) & (y.upper >= 0)
        zero = (x.lower == 0) & (x.upper == 0) & ~invalid
        lower, upper = torch.where(zero, 0., lower), torch.where(zero, 0., upper)
        return self._finish(torch.where(invalid, -torch.inf, lower), torch.where(invalid, torch.inf, upper))

    def square(self, x):
        x = self.interval(x)
        self.charge(2 * x.lower.numel())
        high = torch.maximum(x.lower.abs(), x.upper.abs())
        low = torch.minimum(x.lower.abs(), x.upper.abs())
        crosses = (x.lower <= 0) & (x.upper >= 0)
        lower = torch.where(crosses, 0., _down(low * low).clamp_min(0))
        upper = torch.where(high == 0, 0., _up(high * high))
        return self._finish(lower, upper)

    def sqrt(self, x):
        x = self.interval(x)
        self.charge(2 * x.lower.numel())
        lower = _down(torch.sqrt(x.lower.clamp_min(0))).clamp_min(0)
        upper = torch.where(x.upper == 0, 0., _up(torch.sqrt(x.upper.clamp_min(0))))
        invalid = x.lower < 0
        return self._finish(torch.where(invalid, -torch.inf, lower), torch.where(invalid, torch.inf, upper))

    def total(self, x, dim=-1, *, keepdim=False):
        x = self.interval(x)
        if x.lower.ndim == 0:
            return x
        dim %= x.lower.ndim
        lower, upper = x.lower.movedim(dim, -1), x.upper.movedim(dim, -1)
        if lower.shape[-1] == 0:
            result = GpuInterval.point(torch.zeros(lower.shape[:-1], dtype=torch.float64, device=self.device))
        else:
            while lower.shape[-1] > 1:
                pairs = lower.shape[-1] // 2
                merged = self.add(GpuInterval(lower[..., :2*pairs:2], upper[..., :2*pairs:2]),
                                  GpuInterval(lower[..., 1:2*pairs:2], upper[..., 1:2*pairs:2]))
                if lower.shape[-1] % 2:
                    lower = torch.cat((merged.lower, lower[..., -1:]), dim=-1)
                    upper = torch.cat((merged.upper, upper[..., -1:]), dim=-1)
                else:
                    lower, upper = merged.lower, merged.upper
            result = GpuInterval(lower[..., 0], upper[..., 0])
        return result.unsqueeze(dim) if keepdim else result

    def dot(self, x, y, dim=-1, *, keepdim=False):
        return self.total(self.mul(x, y), dim=dim, keepdim=keepdim)

    def norm(self, x, dim=-1, *, keepdim=False):
        squared = self.total(self.square(x), dim=dim, keepdim=keepdim)
        # Nonnegative summands have a nonnegative exact sum.
        return self.sqrt(GpuInterval(squared.lower.clamp_min(0), squared.upper))

    def _ln2(self):
        return GpuInterval(self._constant(_LN2_LOWER).lower, self._constant(_LN2_UPPER).upper)

    def _scale_power_two(self, x, exponent):
        """Exact normal powers assembled as CUDA integer bits, then enclosed mul.

        Two factors avoid torch.ldexp implementations that construct pow(2,n)
        first (overflowing at n=1024 before multiplication). All exponents here
        lie in [-1478,1478], so both factors are finite normal doubles.
        """
        self.charge(2 * exponent.numel())
        first = exponent.clamp(-512, 512)
        second = exponent - first
        def factor(power):
            bits = ((power.to(torch.int64) + 1023) << 52).contiguous()
            return GpuInterval.point(bits.view(torch.float64))
        return self.mul(self.mul(x, factor(first)), factor(second))

    def _exp_endpoint(self, value):
        self.charge(2 * value.numel())
        safe = value.clamp(-1024., 1024.)
        exponent = torch.round(safe / _LN2_LOWER).to(torch.int64)
        reduced = self.sub(GpuInterval.point(safe), self.mul(GpuInterval.point(exponent.to(torch.float64)), self._ln2()))
        radius = torch.maximum(reduced.lower.abs(), reduced.upper.abs())
        term, polynomial = self._constant(1), self._constant(1)
        # |r| <= 1/2: exp(r) remainder after degree18 is bounded by
        # 2*|r|**19/19!, since exp(1/2)<2. Every polynomial op is enclosed.
        for degree in range(1, 19):
            term = self.div(self.mul(term, reduced), self._constant(degree))
            polynomial = self.add(polynomial, term)
        magnitude = GpuInterval.point(torch.maximum(term.lower.abs(), term.upper.abs()))
        remainder = self.mul(self.div(self.mul(magnitude, GpuInterval.point(radius)), self._constant(19)), self._constant(2))
        bounded = self._finish(self.sub(polynomial, remainder).lower.clamp_min(0), self.add(polynomial, remainder).upper)
        result = self._scale_power_two(bounded, exponent)
        lower, upper = result.lower.clamp_min(0), result.upper
        # exp(-1024)<2**-1074 and exp(1024)>max_binary64, proven already by
        # 1/2<ln(2)<7/10. Large inputs produce honest extended enclosures.
        lower = torch.where(value < -1024., 0., torch.where(value > 1024., _LARGEST, lower))
        upper = torch.where(value < -1024., _SMALLEST, torch.where(value > 1024., torch.inf, upper))
        invalid = (radius > .5) | torch.isnan(value)
        lower, upper = torch.where(invalid, 0., lower), torch.where(invalid, torch.inf, upper)
        zero = value == 0
        return self._finish(torch.where(zero, 1., lower), torch.where(zero, 1., upper))

    def exp(self, x):
        x = self.interval(x)
        endpoints = self._exp_endpoint(torch.stack((x.lower, x.upper)))
        return self._finish(endpoints.lower[0], endpoints.upper[1])

    def _log_endpoint(self, value):
        self.charge(2 * value.numel())
        safe = torch.where((value > 0) & torch.isfinite(value), value, 1.)
        mantissa, exponent = torch.frexp(safe)
        m = GpuInterval.point(mantissa * 2)  # exact binary scaling
        exponent = exponent - 1
        y = self.div(self.sub(m, self._constant(1)), self.add(m, self._constant(1)))
        q = self.square(y)
        term, series = y, y
        for index in range(1, 20):
            term = self.mul(term, q)
            series = self.add(series, self.div(term, self._constant(2 * index + 1)))
        # log(m)=2*atanh(y); 0<=y<=1/3. The omitted positive terms are
        # <=2*y**41/(41*(1-y*y)). Use a nonnegative bound to avoid a tiny
        # outward negative endpoint at the exact m=1 boundary.
        tail = self.div(self.mul(self.mul(term, q), self._constant(2)),
                        self.mul(self._constant(41), self.sub(self._constant(1), q)))
        series = self.mul(series, self._constant(2))
        series = self._finish(series.lower, self.add(series, GpuInterval.point(tail.upper.clamp_min(0))).upper)
        result = self.add(series, self.mul(GpuInterval.point(exponent.to(torch.float64)), self._ln2()))
        lower, upper = result.lower, result.upper
        lower = torch.where(value == 0, -torch.inf, lower)
        upper = torch.where(value == 0, -torch.inf, upper)
        lower = torch.where(torch.isposinf(value), _LARGEST, lower)
        upper = torch.where(torch.isposinf(value), torch.inf, upper)
        invalid = (value < 0) | torch.isnan(value)
        lower, upper = torch.where(invalid, -torch.inf, lower), torch.where(invalid, torch.inf, upper)
        unit = value == 1
        return self._finish(torch.where(unit, 0., lower), torch.where(unit, 0., upper))

    def log(self, x):
        x = self.interval(x)
        endpoints = self._log_endpoint(torch.stack((x.lower, x.upper)))
        invalid = (x.lower < 0) | (x.upper <= 0)
        return self._finish(torch.where(invalid, -torch.inf, endpoints.lower[0]),
                            torch.where(invalid, torch.inf, endpoints.upper[1]))

    def logsumexp(self, x, dim=-1, *, keepdim=False):
        x = self.interval(x)
        if x.lower.shape[dim] == 0:
            raise ValueError("Empty logsumexp is outside the finite-law contract")
        anchor = GpuInterval.point(x.upper.amax(dim, keepdim=True))
        shifted = self.sub(x, anchor)
        result = self.add(anchor, self.log(self.total(self.exp(shifted), dim=dim, keepdim=True)))
        if keepdim:
            return result
        return GpuInterval(result.lower.squeeze(dim), result.upper.squeeze(dim))

    def softmax(self, x, dim=-1):
        x = self.interval(x)
        if x.lower.shape[dim] == 0:
            raise ValueError("Empty softmax is outside the finite-law contract")
        anchor = GpuInterval.point(x.upper.amax(dim, keepdim=True))
        weights = self.exp(self.sub(x, anchor))
        normalizer = self.total(weights, dim=dim, keepdim=True)
        result = self.div(weights, normalizer)
        return self._finish(result.lower.clamp(0, 1), result.upper.clamp(0, 1))

    def softplus(self, x):
        x = self.interval(x)
        # Evaluate monotone endpoint functions separately; the stable identity
        # max(v,0)+log(1+exp(-abs(v))) uses only already-enclosed operations.
        endpoints = torch.stack((x.lower, x.upper))
        positive = GpuInterval.point(endpoints.clamp_min(0))
        residual = self.log(self.add(self._constant(1), self.exp(GpuInterval.point(-endpoints.abs()))))
        result = self.add(positive, residual)
        return self._finish(result.lower[0].clamp_min(0), result.upper[1])
