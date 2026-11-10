# Release procedure

## License and citation metadata

The project uses AGPL-3.0-only. Each release must include the full license text
in `LICENSE`, matching `project.license` metadata in `pyproject.toml`, and the
copyright notices in the source files. The publication preflight checks for
license text and metadata before building distributions.

License files included in wheels are configured in `pyproject.toml`. They include
`LICENSE`, `THIRD_PARTY_NOTICES.md`, and `third_party_licenses/*.txt`.

Before release, verify the author list in `CITATION.cff` and add the release
version and date. Archive releases with Zenodo and record the software DOI;
add the associated article DOI when available. For a paper archive, verify the
software revision against the analysis environment and retain the original
results and artifacts. The current analysis record is described in
[reproducibility](reproducibility.md).

Comparison packages and data need versioned sources or documented access
conditions. The optional LLS reference package is distributed separately.

## Publishing configuration

Public releases use GitHub Actions and PyPI Trusted Publishing. The GitHub
repository must be accessible to users of the published software. Configure
GitHub environments named `testpypi` and `pypi`, with release-tag restrictions
and approval protection for production uploads.

Register a Trusted Publisher on each package index with these settings:

| Setting | PyPI | TestPyPI |
| --- | --- | --- |
| Project | `kinepet` | `kinepet` |
| GitHub owner | `c-salomonsen` | `c-salomonsen` |
| Repository | `KinePET` | `KinePET` |
| Workflow filename | `publish.yml` | `publish.yml` |
| Environment | `pypi` | `testpypi` |

For a new project, register a pending publisher and verify name availability.
The workflow uses GitHub's publishing identity without a stored upload token.
See [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

## Prepare a version

1. Set the package version in `pyproject.toml`, for example `0.4.1`.
2. Regenerate `uv.lock` with `uv lock`.
3. Update the changelog and citation metadata with the release version and date.
4. Run the checks below and commit the release changes.
5. Create the matching Git tag, for example `v0.4.1`.

Previously published tags and artifacts must retain their original contents.
The release preflight requires the tag to match the package version and rejects
development versions for production PyPI uploads.

## Validate distributions

Run the tests described in [development](development.md), then build into a
fresh output directory:

```bash
uv run --group docs mkdocs build --strict
uv run python scripts/check_release.py --tag v0.4.1
uv build --out-dir dist/0.4.1
uv run python scripts/check_distributions.py dist/0.4.1 --require-license
uvx twine check dist/0.4.1/*
```

Substitute the intended version in the tag and output path. Inspect archive
contents, license metadata, and bundled binary dependencies. Identify the OpenMP
runtime and version bundled in each platform wheel. The stored runtime license
texts were copied from GCC 14.2.0 and LLVM 19.1.7; check them against the bundled
versions and update notices when necessary, including after build-image changes.
Any additional library introduced by wheel repair needs its applicable notices.

Build a wheel from the source archive. Install it into a clean environment
outside the checkout and run `scripts/wheel_smoke.py` against that installation.

CI builds CPython 3.11–3.13 wheels on Linux, Windows, and macOS and runs numerical
installed-wheel checks. Publication requires successful Python and C++ tests,
CLI checks, a documentation build, distribution builds, archive checks, and
release metadata validation.

## TestPyPI and production uploads

The workflow supports manual dispatch with `repository=testpypi` or
`repository=pypi` on a version tag. Pushing a `v*` tag also starts the production
workflow; configure production approval protection before pushing the tag.
Approve the production upload only after the TestPyPI checks pass.

For a TestPyPI check, install NumPy from PyPI, then install the exact KinePET
version from TestPyPI with `--no-deps`. Run the installed-wheel check in that
environment. This keeps unrelated dependencies on their normal package index.

After production publication, verify a clean `pip install kinepet==VERSION`,
create the GitHub release, archive it, and update citation links to the versioned
record. Package versions must not be reused for different distribution contents.

## Documentation hosting

The documentation URL is `https://kinepet.salodev.no/`. A hosting change requires
coordinated updates to `site_url` in `mkdocs.yml`, `docs/CNAME`, project URLs, and
redirects from the previous address.
