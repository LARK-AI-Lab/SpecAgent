"""Paper-metric helpers for batch-1-equivalent model-memory read traffic.

The paper reports estimated HBM read only, never write, total, measured HBM, or bandwidth.
The estimate assumes an isolated batch-1 request; concurrent serving can share weight reads,
so per-request sums are not measured DRAM traffic. Write stays unknown (not zero).
"""

from __future__ import annotations

from specagent.protocol.accounting import (
    QuantityStatus,
    ResourceQuantity,
    unknown_quantity,
)

SI_GB = 1.0e9
PAPER_HBM_WRITE_PROVENANCE = (
    "paper_excludes_hbm_write;unknown;not_zero;"
    "not_measured_hbm;not_hbm_bandwidth;not_paper_main_table"
)


def paper_write_unknown() -> ResourceQuantity:
    """HBM write is excluded from the paper metric and must not be filled with 0."""
    return unknown_quantity(provenance=PAPER_HBM_WRITE_PROVENANCE)


def _bounds(quantity: ResourceQuantity) -> tuple[int | None, int | None]:
    if quantity.status in {QuantityStatus.UNKNOWN, QuantityStatus.NOT_APPLICABLE}:
        return None, None
    if quantity.value is not None:
        value = int(quantity.value)
        return value, value
    lower = int(quantity.observed_lower) if quantity.observed_lower is not None else None
    upper = int(quantity.observed_upper) if quantity.observed_upper is not None else None
    return lower, upper


def bytes_to_gb(quantity: ResourceQuantity) -> ResourceQuantity:
    """Convert byte quantities to decimal GB. Status is preserved; unknown stays unknown."""
    extra = f"si_gb={SI_GB:g};parent={quantity.provenance}"
    if quantity.status in {QuantityStatus.UNKNOWN, QuantityStatus.NOT_APPLICABLE}:
        return unknown_quantity(provenance=f"gb_from_unknown;{extra}")
    if quantity.status is QuantityStatus.INCOMPLETE:
        lower, upper = _bounds(quantity)
        return ResourceQuantity(
            value=None,
            status=QuantityStatus.INCOMPLETE,
            provenance=extra,
            coverage=quantity.coverage,
            observed_subtotal=(
                None
                if quantity.observed_subtotal is None
                else float(quantity.observed_subtotal) / SI_GB
            ),
            observed_lower=None if lower is None else float(lower) / SI_GB,
            observed_upper=None if upper is None else float(upper) / SI_GB,
        )
    if quantity.value is None:
        return unknown_quantity(provenance=f"gb_from_missing_value;{extra}")
    return ResourceQuantity(
        value=float(quantity.value) / SI_GB,
        status=quantity.status,
        provenance=extra,
        coverage=quantity.coverage,
    )
