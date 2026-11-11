# SPDX-FileCopyrightText: 2026 Christian Salomonsen
# SPDX-License-Identifier: AGPL-3.0-only

"""Release notes must describe exactly the version being published."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "check_release", Path(__file__).resolve().parents[1] / "scripts/check_release.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
release_notes = _module.release_notes


def test_notes_preserve_markdown_and_exclude_other_versions():
    changelog = """# Changelog

## Unreleased
Future work.

## 0.4.10 (2026-11-12)
A different version.

## 0.4.1 (2026-11-11)
### Added

- Typed models.
- Keep `irr` and `rev`.

## 0.4.0
Older changes.
"""
    assert release_notes(changelog, "0.4.1") == (
        "### Added\n\n- Typed models.\n- Keep `irr` and `rev`.\n"
    )


@pytest.mark.parametrize("changelog", [
    "# Changelog\n## \nMissing heading.\n",
    "# Changelog\n## 0.4.10\nWrong version.\n",
    "# Changelog\n## 0.4.1\n\n## 0.4.0\nOlder.\n",
    "# Changelog\n## 0.4.1\nFirst.\n## 0.4.1\nDuplicate.\n",
])
def test_missing_empty_or_duplicate_release_fails(changelog):
    with pytest.raises(ValueError, match="one nonempty changelog section"):
        release_notes(changelog, "0.4.1")
