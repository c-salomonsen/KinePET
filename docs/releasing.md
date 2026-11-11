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
results and artifacts. Paper-specific analysis provenance is maintained in the companion
`kinepet-paper-validation` repository.

Comparison packages and data need versioned sources or documented access
conditions. The optional LLS reference package is distributed separately.

## Publishing configuration

The GitHub Actions workflow builds Linux, Windows, and macOS wheels and a source
archive, runs validation, then uploads to official PyPI or TestPyPI. Authentication
uses Trusted Publishing (OIDC); no API token, password, or `.pypirc` file is
needed by the workflow. Local Twine credentials are independent of GitHub Actions.

### GitHub environments

In the repository's **Settings → Environments**, create `testpypi` and `pypi`.
For each environment, select **Selected branches and tags**, then add a **Tag**
rule matching `v*`. Branch rules and tag rules are separate.

For `pypi`, enable **Required reviewers** and select the release maintainer.
A sole maintainer who triggers and approves releases must leave **Prevent
self-review** disabled. Save the protection rules before pushing any release tag:
tag pushes automatically start a production publishing run. The environment
approval holds the upload while the release is checked on TestPyPI.

An environment with another name, such as `publish-ci`, does not supply the
settings for `pypi` or `testpypi`. No environment secrets are required for OIDC.
See [GitHub's environment setup](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments).

### PyPI and TestPyPI publishers

Create an account on each index being used. TestPyPI has separate accounts and
publisher configuration. For a new project, open **Publishing** in the account
settings and add a pending GitHub publisher. For an existing project, add the
publisher under the project's publishing settings.

Register a Trusted Publisher on each package index with these settings:

| Setting | PyPI | TestPyPI |
| --- | --- | --- |
| Project | `kinepet` | `kinepet` |
| GitHub owner | `c-salomonsen` | `c-salomonsen` |
| Repository | `KinePET` | `KinePET` |
| Workflow filename | `publish.yml` | `publish.yml` |
| Environment | `pypi` | `testpypi` |

A pending publisher creates the project on the first successful upload; it does
not reserve the name in advance. The owner, repository, workflow filename, and
environment must match the workflow exactly. See
[creating a project with a Trusted Publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and [publishing with OIDC](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

### Local Twine credentials

`~/.pypirc` is used only for uploads run from that computer. It is not available
on GitHub-hosted runners and should not be copied into the repository or workflow.
For token authentication, the conventional username is literally `__token__`
(with two underscores on each side), and the password is the complete API token.
The `[pypi]` section selects official PyPI for `twine upload --repository pypi`.
These credentials are optional when releases use the OIDC workflow above.

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
uvx twine check --strict dist/0.4.1/*
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

Both manual dispatch and tag pushes require a tag matching `pyproject.toml`,
for example `v0.4.1`. Pushing a `v*` tag starts production publishing; the `pypi`
environment must already have its approval rule configured. A push to `main`
runs tests and distribution builds but does not publish a package.

Push the reviewed workflow changes to `main` before using manual dispatch.
Each release run uses the workflow and source at its selected tag. Updating
`main` does not update an existing tag. Never move an archived release tag to
pick up workflow changes; use a new version when the tagged workflow or package
contents need to change.

For an existing tag whose workflow already supports official PyPI and OIDC,
configure the publishers and dispatch that tag directly. Using GitHub CLI:

```bash
gh workflow run publish.yml --repo c-salomonsen/KinePET --ref v0.4.1 -f repository=testpypi
```

The workflow rebuilds the distributions, runs tests and metadata/license checks,
and uploads to TestPyPI. Inspect the run under **Actions → Publish Distributions**.

Verify installation in a fresh environment outside the checkout:

```bash
uv venv /tmp/kinepet-testpypi --python 3.13
uv pip install --python /tmp/kinepet-testpypi/bin/python numpy
uv pip install --python /tmp/kinepet-testpypi/bin/python --index-url https://test.pypi.org/simple/ --no-deps kinepet==0.4.1
```

Run `scripts/wheel_smoke.py` from the selected release against that installed
package. On Windows, substitute a suitable temporary path and the environment's
`Scripts/python.exe`. Installing NumPy separately keeps dependencies on their
normal package index.

After verification, either approve the production run already waiting from the
tag push, or dispatch one production run explicitly:

```bash
gh workflow run publish.yml --repo c-salomonsen/KinePET --ref v0.4.1 -f repository=pypi
```

Approve its `pypi` deployment when prompted. Do not start another production run
if one for that version is already waiting. The production dispatch rebuilds and
checks artifacts; it does not promote the files uploaded to TestPyPI. Published
filenames cannot be overwritten, and the workflow does not silently skip them.

After production publication, verify a clean `pip install kinepet==VERSION`,
create the GitHub release, archive it, and update citation links to the versioned
record. Package versions must not be reused for different distribution contents.

## Documentation hosting

The documentation URL is `https://kinepet.salodev.no/`. A hosting change requires
coordinated updates to `site_url` in `mkdocs.yml`, `docs/CNAME`, project URLs, and
redirects from the previous address.
