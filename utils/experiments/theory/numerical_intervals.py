"""Directed Decimal enclosures for fixed inputs, not empirical error bars.

Basic operations use directed contexts. Decimal exp/ln/sqrt are correctly
rounded to nearest by Python's decimal contract regardless of context rounding;
one adjacent representable Decimal on each side therefore encloses their exact
value. This module makes no claim about construction of its supplied inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN
import math


@dataclass(frozen=True)
class Interval:
    lower: Decimal
    upper: Decimal

    def __post_init__(self):
        if not self.lower.is_finite() or not self.upper.is_finite() or self.lower > self.upper:
            raise ValueError("Intervals require ordered finite endpoints")

    @classmethod
    def point(cls, value):
        if isinstance(value, Decimal):
            number = value
        elif isinstance(value, int):
            number = Decimal(value)
        else:
            number = Decimal.from_float(float(value))
        return cls(number, number)

    @property
    def sign(self):
        if self.lower > 0:
            return "positive"
        if self.upper < 0:
            return "negative"
        if self.lower == self.upper == 0:
            return "zero"
        return "unresolved"

    def floats(self):
        """Outward binary64 serialization, including underflow and overflow."""
        lo, hi = float(self.lower), float(self.upper)
        if math.isfinite(lo) and Decimal.from_float(lo) > self.lower:
            lo = math.nextafter(lo, -math.inf)
        elif lo == math.inf:
            lo = math.nextafter(lo, -math.inf)
        if math.isfinite(hi) and Decimal.from_float(hi) < self.upper:
            hi = math.nextafter(hi, math.inf)
        elif hi == -math.inf:
            hi = math.nextafter(hi, math.inf)
        return lo, hi


class ArithmeticBudgetExceeded(RuntimeError):
    pass


class Directed:
    """Per-row arithmetic context with an explicit deterministic work cap."""
    def __init__(self, precision=64, max_products=2_000_000):
        if precision < 32 or max_products < 1:
            raise ValueError("Invalid directed-arithmetic policy")
        self.precision, self.max_products, self.products = precision, max_products, 0
        self.evaluated_nodes = 0
        options = dict(prec=precision, Emin=-999999999, Emax=999999999)
        self.down = Context(rounding=ROUND_FLOOR, **options)
        self.up = Context(rounding=ROUND_CEILING, **options)
        self.near = Context(rounding=ROUND_HALF_EVEN, **options)

    def charge(self, count=1):
        if self.products + count > self.max_products:
            raise ArithmeticBudgetExceeded("fixed per-row Decimal operation budget exhausted")
        self.products += count

    def add(self, x, y):
        self.charge()
        return Interval(self.down.add(x.lower, y.lower), self.up.add(x.upper, y.upper))

    def neg(self, x):
        return Interval(x.upper.copy_negate(), x.lower.copy_negate())

    def sub(self, x, y):
        return self.add(x, self.neg(y))

    def mul(self, x, y):
        self.charge(4)
        pairs = [(a, b) for a in (x.lower, x.upper) for b in (y.lower, y.upper)]
        return Interval(min(self.down.multiply(a, b) for a, b in pairs),
                        max(self.up.multiply(a, b) for a, b in pairs))

    def div(self, x, y):
        if y.lower <= 0 <= y.upper:
            raise ArithmeticError("Division interval contains zero")
        self.charge(4)
        pairs = [(a, b) for a in (x.lower, x.upper) for b in (y.lower, y.upper)]
        return Interval(min(self.down.divide(a, b) for a, b in pairs),
                        max(self.up.divide(a, b) for a, b in pairs))

    def square(self, x):
        self.charge(2)
        high = max(x.lower.copy_abs(), x.upper.copy_abs())
        low = Decimal(0) if x.lower <= 0 <= x.upper else min(x.lower.copy_abs(), x.upper.copy_abs())
        return Interval(self.down.multiply(low, low), self.up.multiply(high, high))

    def _transcendental(self, method, x, *, nonnegative=False):
        self.charge(2)
        lo = getattr(self.near, method)(x.lower)
        hi = getattr(self.near, method)(x.upper)
        lo, hi = self.near.next_minus(lo), self.near.next_plus(hi)
        if nonnegative:
            lo = max(Decimal(0), lo)
        return Interval(lo, hi)

    def exp(self, x):
        if x.lower == x.upper == 0:
            return Interval.point(1)
        return self._transcendental("exp", x, nonnegative=True)

    def log(self, x):
        if x.lower <= 0:
            raise ArithmeticError("Log interval includes a nonpositive value")
        if x.lower == x.upper == 1:
            return Interval.point(0)
        return self._transcendental("ln", x)

    def sqrt(self, x):
        if x.lower < 0:
            raise ArithmeticError("Square-root interval includes negative values")
        if x.lower == x.upper == 0:
            return Interval.point(0)
        return self._transcendental("sqrt", x, nonnegative=True)

    def total(self, values):
        result = Interval.point(0)
        for value in values:
            result = self.add(result, value)
        return result

    def dot(self, x, y):
        if len(x) != len(y):
            raise ValueError("Dot-product lengths differ")
        return self.total(self.mul(a, b) for a, b in zip(x, y))

    def norm(self, x):
        return self.sqrt(self.total(self.square(a) for a in x))

    def logsumexp(self, values):
        if not values:
            raise ValueError("Empty complement has undefined log odds")
        anchor = Interval.point(max(value.upper for value in values))
        return self.add(anchor, self.log(self.total(self.exp(self.sub(value, anchor)) for value in values)))

    def softmax(self, values):
        anchor = Interval.point(max(value.upper for value in values))
        unnormalized = [self.exp(self.sub(value, anchor)) for value in values]
        normalizer = self.total(unnormalized)
        return [self.div(value, normalizer) for value in unnormalized]

    def softplus(self, x):
        def endpoint(value):
            point = Interval.point(value)
            if value >= 0:
                return self.add(point, self.log(self.add(Interval.point(1), self.exp(self.neg(point)))))
            return self.log(self.add(Interval.point(1), self.exp(point)))
        lo, hi = endpoint(x.lower), endpoint(x.upper)
        return Interval(lo.lower, hi.upper)


def interval_gain(arithmetic, intercept, slopes, target_atom):
    """Enclose gains of ONE affine logit segment, never absolute-odds subtraction."""
    c = arithmetic
    if len(intercept) != len(slopes) or not 0 <= target_atom < len(intercept):
        raise ValueError("Gain dimensions/target differ")
    if len(intercept) == 1:
        return {"H": Interval.point(0), "G": None, "gain_sign": "zero",
                "slope0": Interval.point(0), "slope1": Interval.point(0)}
    b = [c.sub(value, intercept[target_atom]) for i, value in enumerate(intercept) if i != target_atom]
    a = [c.sub(value, slopes[target_atom]) for i, value in enumerate(slopes) if i != target_atom]
    if all(value.lower == value.upper == 0 for value in a):
        return {"H": Interval.point(0), "G": Interval.point(0), "gain_sign": "zero",
                "slope0": Interval.point(0), "slope1": Interval.point(0)}
    weights = c.softmax(b)
    # Center slopes before exponentiation; their maximum is a fixed shift.
    shift = Interval.point(max(value.upper for value in a))
    expectation = c.total(c.mul(weight, c.exp(c.sub(slope, shift))) for weight, slope in zip(weights, a))
    G = c.neg(c.add(shift, c.log(expectation)))
    complement0 = c.logsumexp(b)
    complement1 = c.logsumexp([c.add(x, y) for x, y in zip(b, a)])
    H = c.sub(c.softplus(complement0), c.softplus(complement1))
    weights1 = c.softmax([c.add(x, y) for x, y in zip(b, a)])
    slope0 = c.neg(c.dot(weights, a))
    slope1 = c.neg(c.dot(weights1, a))
    return {"H": H, "G": G, "gain_sign": G.sign, "slope0": slope0, "slope1": slope1}


def negative_condition_precheck(D, ec, eu=None, *, arithmetic=None):
    """V>=0 and eu>=0: a negative upper bound resolves sign, not magnitude."""
    c = arithmetic or Directed()
    eu = Interval.point(0) if eu is None else eu
    upper = c.sub(c.sub(D, ec), eu).upper
    return {"upper": upper, "condition_sign_status": "negative" if upper < 0 else "unresolved",
            "condition_value_status": "unavailable_variation_not_computed"}


def support_geometry(arithmetic, atoms):
    """Bounding-box diagonal encloses diameter without quadratic atom pairs."""
    c = arithmetic
    if not atoms or not atoms[0] or any(len(atom) != len(atoms[0]) for atom in atoms):
        raise ValueError("Invalid finite support shape")
    width = [c.sub(Interval.point(max(atom[k].upper for atom in atoms)),
                   Interval.point(min(atom[k].lower for atom in atoms))) for k in range(len(atoms[0]))]
    diameter = c.norm(width)
    maximum_norm = max(c.norm(atom).upper for atom in atoms)
    return diameter, Interval.point(maximum_norm)


def posterior_mean(arithmetic, atoms, weights):
    c = arithmetic
    return [c.total(c.mul(weight, atom[k]) for weight, atom in zip(weights, atoms))
            for k in range(len(atoms[0]))]


def enclose_variation(arithmetic, atoms, intercept, slopes, current_mean, D, ec, eu,
                      *, max_nodes=65, absolute_width=1e-6, current_in_convex_hull=True,
                      current_mass_defect=None, target_atom=0):
    """Certified midpoint enclosure including every node operation.

    For a_j affine logit slopes, |d mean/ds| <= diameter*range(a)/4:
    Cauchy--Schwarz bounds each directional covariance, and Popoviciu bounds
    both variances. This also covers rounded cached slopes whose exact vector
    construction is unavailable. If slopes are beta<h,u_j>, this recovers the
    requested beta*||h||*diameter**2/4 bound. Sum L*w**2/4 over leaves.
    """
    if not 0 <= target_atom < len(slopes):
        raise ValueError("Invalid target slope index")
    if max_nodes < 1 or absolute_width <= 0:
        raise ValueError("Variation budget requires a positive node count")
    c = arithmetic
    diameter, maximum_norm = support_geometry(c, atoms)
    spread = c.sub(Interval.point(max(value.upper for value in slopes)),
                   Interval.point(min(value.lower for value in slopes)))
    lipschitz = c.div(c.mul(diameter, spread), Interval.point(4))
    cap = diameter.upper
    if not current_in_convex_hull:
        if current_mass_defect is None:
            raise ValueError("Rounded current weights require an explicit mass-defect bound")
        cap = c.add(diameter, c.mul(current_mass_defect, maximum_norm)).upper
    gain_lipschitz = c.div(c.square(spread), Interval.point(4))
    nodes = 0

    def leaf(left, right):
        nonlocal nodes
        width = c.sub(Interval.point(right), Interval.point(left))
        midpoint = c.div(c.add(Interval.point(left), Interval.point(right)), Interval.point(2))
        weights = c.softmax([c.add(b, c.mul(midpoint, a)) for b, a in zip(intercept, slopes)])
        mean = posterior_mean(c, atoms, weights)
        f = c.norm([c.sub(a, b) for a, b in zip(mean, current_mean)])
        midpoint_integral = c.mul(width, f)
        remainder = c.div(c.mul(lipschitz, c.square(width)), Interval.point(4))
        # d log p_target / ds = a_target - E_p[a], including a nonzero
        # common logit gauge. Production target-relative payloads use a_target=0.
        derivative = c.sub(slopes[target_atom], c.dot(weights, slopes))
        gain_midpoint = c.mul(width, derivative)
        gain_remainder = c.div(c.mul(gain_lipschitz, c.square(width)), Interval.point(4))
        gain_interval = Interval(c.sub(gain_midpoint, gain_remainder).lower,
                                 c.add(gain_midpoint, gain_remainder).upper)
        nodes += 1
        c.evaluated_nodes += 1
        lo = max(Decimal(0), c.sub(midpoint_integral, remainder).lower)
        hi = min(c.mul(width, Interval.point(cap)).upper, c.add(midpoint_integral, remainder).upper)
        if hi < lo:
            raise ArithmeticError("Inconsistent node enclosure and convex-hull cap")
        return (left, right, Interval(lo, hi), gain_interval)

    leaves = [leaf(Decimal(0), Decimal(1))]
    stop = "node_budget_exhausted"
    while True:
        V = c.total(value for _, _, value, _ in leaves)
        integrated_H = c.total(value for _, _, _, value in leaves)
        V = Interval(max(Decimal(0), V.lower), min(cap, V.upper))
        margin = c.sub(c.sub(c.sub(D, ec), eu), V)
        width = c.sub(Interval.point(V.upper), Interval.point(V.lower)).upper
        if margin.sign in {"positive", "negative", "zero"}:
            stop = "condition_sign_resolved"
            break
        if width <= Decimal.from_float(float(absolute_width)):
            stop = "requested_variation_width_reached"
            break
        if nodes + 2 > max_nodes:
            break
        # Exact Decimal endpoint subtraction for ranking need not be enclosed;
        # it chooses work only, never changes a reported enclosure.
        index = max(range(len(leaves)), key=lambda i: c.up.subtract(leaves[i][2].upper, leaves[i][2].lower))
        left, right, _, _ = leaves.pop(index)
        midpoint = c.near.divide(c.near.add(left, right), Decimal(2))
        leaves.extend([leaf(left, midpoint), leaf(midpoint, right)])
    return {"V": V, "M": margin, "nodes": nodes, "stopping_reason": stop,
            "lipschitz": lipschitz, "diameter": diameter, "integrated_H": integrated_H,
            "error_method": "directed_decimal_nodes_plus_global_lipschitz_midpoint_remainder"}
