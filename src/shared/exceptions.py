"""Custom exceptions for the Micro-Climate Zoning AI pipeline."""

from __future__ import annotations


class IngestionValidationError(Exception):
    """Raised when a required data stream fails validation during ingestion.

    Attributes:
        stream_name: Name of the data stream that failed (e.g., 'lidar', 'satellite', 'meteorological').
        reason: Human-readable explanation of why validation failed.
        affected_extent: Optional dict describing the spatial extent affected by the failure.
    """

    def __init__(
        self,
        stream_name: str,
        reason: str,
        affected_extent: dict | None = None,
    ) -> None:
        self.stream_name = stream_name
        self.reason = reason
        self.affected_extent = affected_extent
        super().__init__(f"Ingestion validation failed for '{stream_name}': {reason}")


class CheckpointError(Exception):
    """Raised when a PINN checkpoint cannot be loaded.

    This covers both corruption (truncated/damaged file) and architecture mismatch
    (checkpoint was saved with a different model configuration). A partially
    initialized model must NEVER be returned — this error is the only outcome
    for invalid checkpoints.

    Attributes:
        path: Path to the checkpoint file that failed to load.
        reason: Human-readable explanation of the incompatibility or corruption.
    """

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Cannot load checkpoint '{path}': {reason}")


class InfeasibilityError(Exception):
    """Raised when optimization constraints are mutually infeasible.

    The optimizer detects this before beginning the MOBO search and provides
    details about which constraints conflict.

    Attributes:
        conflicting_constraints: List of constraint descriptions that are mutually infeasible.
        details: Optional dict with quantitative details about the conflict.
    """

    def __init__(
        self,
        conflicting_constraints: list[str],
        details: dict | None = None,
    ) -> None:
        self.conflicting_constraints = conflicting_constraints
        self.details = details
        msg = "Mutually infeasible constraints: " + "; ".join(conflicting_constraints)
        super().__init__(msg)


class UCMValidationError(Exception):
    """Raised when UCM input fields are out of physically plausible ranges.

    Attributes:
        block_id: The block identifier that failed validation.
        field: The field name that is invalid.
        value: The invalid value.
        valid_range: Description of the valid range.
    """

    def __init__(self, block_id: str, field: str, value: float, valid_range: str) -> None:
        self.block_id = block_id
        self.field = field
        self.value = value
        self.valid_range = valid_range
        super().__init__(
            f"UCM validation failed for block '{block_id}': "
            f"{field}={value} is outside valid range {valid_range}"
        )


class ProvenanceError(Exception):
    """Raised when provenance record operations fail."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
