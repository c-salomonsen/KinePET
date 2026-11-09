# SPDX-FileCopyrightText: 2026 Christian Salomonsen
# SPDX-License-Identifier: AGPL-3.0-only

"""Check archive contents and required notices before publication."""
import argparse
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


NOTICES = ("THIRD_PARTY_NOTICES.md", "third_party_licenses/GCC-GPL-3.0.txt",
           "third_party_licenses/GCC-Runtime-Exception.txt", "third_party_licenses/LLVM-OpenMP.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--require-license", action="store_true")
    args = parser.parse_args()
    archives = sorted(args.directory.glob("*.whl")) + sorted(args.directory.glob("*.tar.gz"))
    if not archives:
        raise SystemExit("No distribution archives found")
    for archive in archives:
        if archive.suffix == ".whl":
            with zipfile.ZipFile(archive) as handle:
                names = handle.namelist()
            required = ["kinepet/__init__.py", "kinepet/py.typed", *NOTICES]
        else:
            with tarfile.open(archive) as handle:
                names = handle.getnames()
            required = ["pyproject.toml", "CMakeLists.txt", "src/validation.hpp",
                        "scripts/wheel_smoke.py", "tests/cpp/test_core.cpp", *NOTICES]
        if args.require_license:
            required.append("LICENSE")
        for suffix in required:
            if not any(name == suffix or name.endswith("/" + suffix) for name in names):
                raise SystemExit(f"{archive.name}: missing {suffix}")
        for name in names:
            parts = PurePosixPath(name).parts
            if any(part in {"data", "output", "notebooks", ".venv", "__pycache__", "AGENTS.md"} for part in parts):
                raise SystemExit(f"{archive.name}: unwanted member {name}")
        binaries = [name for name in names if any(ext in name for ext in (".so", ".dylib", ".dll", ".pyd"))]
        print(f"Checked {archive.name}; binary inventory: {binaries}")


if __name__ == "__main__":
    main()
