"""JSON schema for zoning directive validation."""

ZONING_DIRECTIVE_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": [
        "block_id",
        "zone_class",
        "max_height_m",
        "height_justification",
        "permitted_uses",
        "mandatory_interventions",
        "compliance_deadline_months",
        "review_trigger",
        "provenance",
    ],
    "properties": {
        "block_id": {"type": "string"},
        "zone_class": {
            "type": "string",
            "enum": [
                "WIND_CORRIDOR_CRITICAL",
                "THERMAL_REMEDIATION",
                "DENSITY_ADAPTIVE",
                "BASELINE_UNCHANGED",
            ],
        },
        "max_height_m": {"type": ["number", "null"]},
        "height_justification": {"type": "string"},
        "permitted_uses": {"type": "array", "items": {"type": "string"}},
        "mandatory_interventions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "cooling_effect_estimate_c"],
                "properties": {
                    "type": {"type": "string"},
                    "cooling_effect_estimate_c": {"type": "number"},
                },
            },
        },
        "compliance_deadline_months": {"type": "integer"},
        "review_trigger": {"type": "string"},
        "provenance": {
            "type": "object",
            "required": [
                "udt_version_id",
                "pinn_checkpoint_version",
                "cfd_sample_ids",
                "mobo_run_id",
            ],
            "properties": {
                "udt_version_id": {"type": "string"},
                "pinn_checkpoint_version": {"type": "string"},
                "cfd_sample_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "mobo_run_id": {"type": "string"},
            },
        },
    },
}
