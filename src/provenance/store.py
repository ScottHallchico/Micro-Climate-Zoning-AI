"""Central provenance store — records and queries artifact lineage across the entire pipeline.

Every component in the pipeline records provenance when it produces an output artifact.
This store is the single source of truth for auditability and traceability: any zoning
directive can be traced back to its source data, model predictions, and optimization run.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.shared.types import ProvenanceRecord
from src.shared.exceptions import ProvenanceError

logger = logging.getLogger(__name__)


class ProvenanceStore:
    """Central provenance store backed by a local JSON file or PostGIS.

    For initial development, this uses a file-backed store. In production,
    this would be replaced with PostGIS table operations.

    Usage:
        store = ProvenanceStore(store_path="./data/provenance.json")
        store.record_artifact(ProvenanceRecord(...))
        lineage = store.get_lineage("artifact-123")
    """

    def __init__(self, store_path: str = "./data/provenance.json") -> None:
        """Initialize the provenance store.

        Args:
            store_path: Path to the JSON file backing the store.
        """
        self._store_path = Path(store_path)
        self._records: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load existing records from the backing store."""
        if self._store_path.exists():
            try:
                with open(self._store_path, "r") as f:
                    data = json.load(f)
                self._records = data.get("records", {})
                logger.info(f"Loaded {len(self._records)} provenance records from {self._store_path}")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not load provenance store from {self._store_path}: {e}")
                self._records = {}

    def _save(self) -> None:
        """Persist records to the backing store."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._store_path, "w") as f:
            json.dump({"records": self._records}, f, indent=2, default=str)

    def record_artifact(self, record: ProvenanceRecord) -> None:
        """Record a new artifact in the provenance store.

        Args:
            record: The provenance record to store.

        Raises:
            ProvenanceError: If the record cannot be stored.
        """
        try:
            record_dict = {
                "artifact_id": record.artifact_id,
                "artifact_type": record.artifact_type,
                "producing_component": record.producing_component,
                "input_artifact_ids": record.input_artifact_ids,
                "creation_timestamp": record.creation_timestamp.isoformat(),
                "metadata": record.metadata,
            }
            self._records[record.artifact_id] = record_dict
            self._save()
            logger.info(
                f"Recorded provenance for artifact '{record.artifact_id}' "
                f"(type={record.artifact_type}, component={record.producing_component})"
            )
        except Exception as e:
            raise ProvenanceError(f"Failed to record artifact '{record.artifact_id}': {e}") from e

    def get_record(self, artifact_id: str) -> ProvenanceRecord | None:
        """Retrieve a single provenance record by artifact ID.

        Args:
            artifact_id: The unique identifier of the artifact.

        Returns:
            The ProvenanceRecord if found, None otherwise.
        """
        record_dict = self._records.get(artifact_id)
        if record_dict is None:
            return None
        return ProvenanceRecord(
            artifact_id=record_dict["artifact_id"],
            artifact_type=record_dict["artifact_type"],
            producing_component=record_dict["producing_component"],
            input_artifact_ids=record_dict["input_artifact_ids"],
            creation_timestamp=datetime.fromisoformat(record_dict["creation_timestamp"]),
            metadata=record_dict.get("metadata", {}),
        )

    def get_lineage(self, artifact_id: str) -> list[ProvenanceRecord]:
        """Retrieve the full lineage chain for an artifact.

        Traces back through all input artifacts recursively to build
        the complete provenance trail from the given artifact to its
        original source data.

        Args:
            artifact_id: The artifact to trace lineage for.

        Returns:
            List of ProvenanceRecords in dependency order (sources first).
        """
        visited: set[str] = set()
        lineage: list[ProvenanceRecord] = []

        def _trace(aid: str) -> None:
            if aid in visited:
                return
            visited.add(aid)
            record = self.get_record(aid)
            if record is None:
                return
            # Trace inputs first (depth-first)
            for input_id in record.input_artifact_ids:
                _trace(input_id)
            lineage.append(record)

        _trace(artifact_id)
        return lineage

    def get_artifacts_by_type(self, artifact_type: str) -> list[ProvenanceRecord]:
        """Retrieve all artifacts of a given type.

        Args:
            artifact_type: The type of artifacts to retrieve (e.g., "udt", "pinn_checkpoint").

        Returns:
            List of matching ProvenanceRecords.
        """
        results = []
        for record_dict in self._records.values():
            if record_dict["artifact_type"] == artifact_type:
                results.append(
                    ProvenanceRecord(
                        artifact_id=record_dict["artifact_id"],
                        artifact_type=record_dict["artifact_type"],
                        producing_component=record_dict["producing_component"],
                        input_artifact_ids=record_dict["input_artifact_ids"],
                        creation_timestamp=datetime.fromisoformat(record_dict["creation_timestamp"]),
                        metadata=record_dict.get("metadata", {}),
                    )
                )
        return results

    def get_artifacts_by_component(self, producing_component: str) -> list[ProvenanceRecord]:
        """Retrieve all artifacts produced by a given component.

        Args:
            producing_component: The component name (e.g., "ingestion_pipeline").

        Returns:
            List of matching ProvenanceRecords.
        """
        results = []
        for record_dict in self._records.values():
            if record_dict["producing_component"] == producing_component:
                results.append(
                    ProvenanceRecord(
                        artifact_id=record_dict["artifact_id"],
                        artifact_type=record_dict["artifact_type"],
                        producing_component=record_dict["producing_component"],
                        input_artifact_ids=record_dict["input_artifact_ids"],
                        creation_timestamp=datetime.fromisoformat(record_dict["creation_timestamp"]),
                        metadata=record_dict.get("metadata", {}),
                    )
                )
        return results

    def clear(self) -> None:
        """Clear all records from the store. Use with caution."""
        self._records = {}
        self._save()
        logger.warning("Provenance store cleared")
