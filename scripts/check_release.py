"""Fail closed before uploading a versioned public release."""
import argparse
from pathlib import Path
import tomllib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repository", choices=("pypi", "testpypi"), default="pypi")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    if args.tag != f"v{version}":
        raise SystemExit(f"Release tag must be v{version}; got {args.tag!r}")
    if args.repository == "pypi" and ".dev" in version:
        raise SystemExit("Choose a release version before publishing to PyPI")
    if project.get("license") not in ("AGPL-3.0-only", "AGPL-3.0-or-later"):
        raise SystemExit("Maintainer must add the chosen AGPLv3 SPDX expression to project.license")
    license_path = root / "LICENSE"
    if not license_path.is_file() or "GNU AFFERO GENERAL PUBLIC LICENSE" not in license_path.read_text():
        raise SystemExit("Maintainer must add the full AGPLv3 license text as LICENSE")
    if "salodev.no" in (root / "uv.lock").read_text():
        raise SystemExit("Release dependency lock must use public package sources")
    print(f"Release metadata checks passed: {version}")


if __name__ == "__main__":
    main()
