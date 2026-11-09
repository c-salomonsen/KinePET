# Public release procedure

## Items the maintainer must complete

1. Add the chosen AGPLv3 text as `LICENSE` and set `project.license` in
   `pyproject.toml` to `AGPL-3.0-only` or `AGPL-3.0-or-later`, consistently with
   the intended grant. Confirm rights-holder notices. The license is deliberately
   left for the maintainer; the publication gate refuses to upload without it.
   `LICENSE*` and `THIRD_PARTY_NOTICES.md` are included in wheels automatically.
2. Confirm the citation author list and the paper revision described in
   [reproducibility](reproducibility.md). Archive the exact paper release with
   Zenodo and add its DOI, release version/date, and eventually the article DOI
   to `CITATION.cff`. The current contributor entity is a provisional attribution.
3. Make any separately required comparison packages/data publicly available or
   document their access conditions. The optional LLS reference is still external.
4. Make the GitHub repository public when ready. Configure GitHub environments
   `testpypi` and `pypi` with release-tag restrictions and required reviewers.
5. Register pending Trusted Publishers on TestPyPI and PyPI: owner
   `c-salomonsen`, repository `KinePET`, workflow `publish.yml`, environment
   `testpypi` or `pypi`, project `kinepet`. Verify name availability on each service.
   No upload token or private package server is used by the new workflow.

## Validate and publish

Keep `v0.4.0` and its artifacts unchanged. This branch uses `0.5.0.dev0` until a
release version is chosen. Update `pyproject.toml`, regenerate `uv.lock`, update
the changelog and citation, and commit before creating a matching `vVERSION` tag.

Run the tests from [development](development.md), then:

```bash
uv run --group docs mkdocs build --strict
uv build
uvx twine check dist/*
```

Use a clean distribution directory to avoid mixing old releases. Inspect source
and wheel contents, licenses, metadata, and bundled binary dependencies. The
notices cover nanobind, robin-map, and commonly bundled OpenMP runtimes; verify
any additional library introduced by platform wheel repair before release.
Identify the OpenMP runtime and version bundled in each platform wheel. The
stored runtime license texts were copied from GCC 14.2.0 and LLVM 19.1.7;
check them against the actual bundled versions and update the notices and texts
when necessary, including after build-image changes.
Install the wheel into a clean environment outside the checkout and run
`scripts/wheel_smoke.py`. Also build a wheel from the sdist.

The distribution workflow builds CPython 3.11–3.13 wheels on Linux, Windows, and
macOS and runs a numerical installed-wheel check. Local success does not replace
those platform results. The publication workflow requires the test suite, CLI
checks, documentation build, distribution builds, matching tag/version, and
license preflight to succeed before upload.

For rehearsal, use a version tag and manually dispatch `Publish Distributions`
with `repository=testpypi`. Install NumPy from PyPI first, then install the exact
KinePET wheel from TestPyPI with `--no-deps` to avoid using TestPyPI as a source
for unrelated dependencies. Run the installed-wheel check. A pushed `v*` tag
selects production PyPI; the protected `pypi` environment should require review.
Development versions are rejected by the production preflight. No version should
be overwritten or reused with different contents.

The workflow follows [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
After publishing, verify a clean `pip install kinepet==VERSION`, create the
GitHub release, archive it, and link the paper to the versioned record.

## Documentation hosting

Documentation continues to use `https://kinepet.salodev.no/`. GitHub Pages is an
optional later migration; update `site_url`, `docs/CNAME`, and any redirects
together when changing the public URL.
