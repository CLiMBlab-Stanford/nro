"""Exercise a shared-release transition through a real one-shot scheduler."""

from pathlib import Path

import pytest

from nro.configuration.definition_migrations import SCHEMA_VERSION
from nro.engine.upgrade_rehearsal import rehearse

pytestmark = pytest.mark.integration


def test_candidate_coordinates_maintenance_against_an_old_active_source() -> None:
    checkout = Path(__file__).resolve().parents[1]

    result = rehearse(checkout)

    assert result["baseline_source"] != result["candidate_source"]
    assert result["definitions_schema"] == SCHEMA_VERSION
