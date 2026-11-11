# SPDX-FileCopyrightText: 2026 Christian Salomonsen
# SPDX-License-Identifier: AGPL-3.0-only

"""Fail closed before uploading a versioned public release."""
import argparse
from pathlib import Path
import re
import tomllib


def release_notes(changelog: str, version: str) -> str:
    """Extract one version's notes without including adjacent releases."""
    sections = re.split(r"^## ", changelog, flags=re.MULTILINE)[1:]
    matches = []
    for section in sections:
        heading, _, body = section.partition("\n")
        if re.match(re.escape(version) + r"(?:\s|$)", heading):
            matches.append(body.strip())
    if len(matches) != 1 or not matches[0]:
        raise ValueError(f"Expected one nonempty changelog section for {version}")
    return matches[0] + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repository", choices=("pypi", "testpypi"), default="pypi")
    parser.add_argument("--notes-output", type=Path, help="Write this version's changelog notes")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    if args.tag != f"v{version}":
        raise SystemExit(f"Release tag must be v{version}; got {args.tag!r}")
    if args.repository == "pypi" and ".dev" in version:
        raise SystemExit("Choose a release version before publishing to PyPI")
    if project.get("license") not in ("AGPL-3.0-only", "AGPL-3.0-or-later"):
        raise SystemExit("Missing or unsupported project.license: expected AGPL-3.0-only or AGPL-3.0-or-later")
    license_path = root / "LICENSE"
    if not license_path.is_file() or "GNU AFFERO GENERAL PUBLIC LICENSE" not in license_path.read_text():
        raise SystemExit("Missing or invalid LICENSE: expected the full AGPLv3 license text")
    if "salodev.no" in (root / "uv.lock").read_text():
        raise SystemExit("Release dependency lock must use public package sources")
    if args.notes_output:
        try:
            notes = release_notes((root / "CHANGELOG.md").read_text(), version)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        args.notes_output.write_text(notes)
    print(f"Release metadata checks passed: {version}")


if __name__ == "__main__":
    main()
