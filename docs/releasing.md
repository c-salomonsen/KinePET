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
archive once. After validation, it uploads to TestPyPI, prepares a draft GitHub
release, and waits for production approval. It then uploads the same files to
official PyPI and publishes the GitHub release. Authentication
uses Trusted Publishing (OIDC); no API token, password, or `.pypirc` file is
needed by the workflow. Local Twine credentials are independent of GitHub Actions.

### GitHub environments

In the repository's **Settings → Environments**, create `testpypi` and `pypi`.
For each environment, select **Selected branches and tags**, then add a **Tag**
rule matching `v*`. Branch rules and tag rules are separate.

For `pypi`, enable **Required reviewers** and select the release maintainer.
A sole maintainer who triggers and approves releases must leave **Prevent
self-review** disabled. Save the protection rules before pushing any release tag:
tag pushes automatically start the release pipeline. The environment
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

1. Set the package version in `pyproject.toml`, for example `0.4.2`.
2. Regenerate `uv.lock` with `uv lock`.
3. Update the changelog and citation metadata with the release version and date.
4. Run the checks below and commit the release changes.
5. Create the matching Git tag, for example `v0.4.2`.

Previously published tags and artifacts must retain their original contents.
The release preflight requires the tag to match the package version and rejects
development versions for production PyPI uploads.

## Validate distributions

Run the tests described in [development](development.md), then build into a
fresh output directory:

```bash
uv run --group docs mkdocs build --strict
uv run python scripts/check_release.py --tag v0.4.2
uv build --out-dir dist/0.4.2
uv run python scripts/check_distributions.py dist/0.4.2 --require-license
uvx twine check --strict dist/0.4.2/*
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
installed-wheel checks. macOS wheels target Apple Silicon (`arm64`) and build on
the `macos-15` runner; Intel macOS wheels are not included in this matrix.
Publication requires successful Python and C++ tests,
CLI checks, a documentation build, distribution builds, archive checks, and
release metadata validation.

## Routine CI and release checks

| Trigger | Checks |
| --- | --- |
| Push to `main` or a pull request | One Linux job on Python 3.13: C++ tests, Python tests, and CLI/API parity |
| Documentation, MkDocs configuration, dependency metadata, or documentation-workflow changes | Strict documentation build without deployment |
| Version tag | Python 3.11–3.13 tests, C++/CLI checks, documentation, source archive, and Linux/Windows/macOS wheels |
| Manual **Tests** run | Full Python matrix by default; a single configuration can be selected |
| Manual **Build Distributions** run | Source archive and all platform wheels, without publication |

The release workflow calls the same test and build workflows used for manual
checks. Ordinary pushes do not build distribution wheels. The **Documentation**
workflow validates pull requests and releases;
the release pipeline deploys its validated documentation artifact after package
and GitHub release publication. **Publish Documentation** is a manual override
with an explicit source ref. Routine documentation checks use path filters; do
not require them to complete on every pull request in branch protection.

If branch protection requires the previous Python or CLI job names, update it to
require `Python and C++ tests (3.13)` after the new workflow has run.

## TestPyPI and production uploads

Push the reviewed workflow changes to `main` before creating the next release
tag. Each release run uses the workflow and source at its selected tag. Updating
`main` does not update an existing tag. Never move an archived release tag to
pick up workflow changes; use a new version when the tagged workflow or package
contents need to change.

For a new version, after completing the preparation and validation steps:

```bash
git tag -a v0.4.2 -m "KinePET 0.4.2"
git push origin v0.4.2
```

Substitute the intended version. The `v*` tag push starts **Publish
Distributions**, which performs these steps:

1. Check the tag, package version, license, and matching changelog section.
2. Run the full tests and documentation build, and build all distributions.
3. Validate the archives and save them with SHA-256 checksums in the
   `release-distributions` Actions artifact, retained for 90 days.
4. Upload those files to TestPyPI through the `testpypi` environment. If that
   environment also has a reviewer rule, approve its deployment first.
5. Create a draft GitHub release using the version's changelog entry. An existing
   draft is reused, preserving any edits to its notes.
6. Wait for approval of the `pypi` environment deployment.
7. Upload the same saved distributions to PyPI, then publish the GitHub release.
8. Deploy the documentation artifact built from that release commit to GitHub
   Pages. This uses the earlier validated build, without rebuilding the site.

The workflow downloads the saved artifact by its immutable artifact ID and
verifies checksums before each upload. There is no rebuild between TestPyPI and
PyPI. Required reviewers on the `pypi` environment provide the manual approval
step; without that protection, the pipeline continues automatically.

While the production job is waiting, review the draft release under the
repository's **Releases** page. Edit and save the draft notes if necessary; leave
publication to the workflow. Verify installation in a fresh environment outside
the checkout. With uv:

```bash
uv venv /tmp/kinepet-testpypi --python 3.13
uv pip install --python /tmp/kinepet-testpypi/bin/python numpy
uv pip install --python /tmp/kinepet-testpypi/bin/python --index-url https://test.pypi.org/simple/ --no-deps kinepet==0.4.2
```

Or with pip:

```bash
python3.13 -m venv /tmp/kinepet-testpypi
/tmp/kinepet-testpypi/bin/python -m pip install numpy
/tmp/kinepet-testpypi/bin/python -m pip install --index-url https://test.pypi.org/simple/ --no-deps kinepet==0.4.2
```

Run `scripts/wheel_smoke.py` from the selected release against that installed
package. On Windows, substitute a suitable temporary path and the environment's
`Scripts/python.exe`. Installing NumPy separately keeps dependencies on their
normal package index.

After verification, open the waiting workflow run, choose **Review deployments**,
and approve `pypi`. Do not start a separate production run. After publication,
verify a clean installation from official PyPI and check the GitHub release and
Zenodo archive. Record the versioned DOI in the citation metadata when available.

### Manual runs and recovery

To start the pipeline manually for a tag that contains this workflow:

```bash
gh workflow run publish.yml --repo c-salomonsen/KinePET --ref v0.4.2
```

Manual runs follow the same TestPyPI → approval → PyPI sequence. Dispatching on a
branch fails the tag check. Do not dispatch another run if a release run for that
version is already waiting or has uploaded packages.

If a job fails before uploading any files, correct the external configuration
where applicable and use **Re-run failed jobs** on the same run. Successful build
and upload jobs remain complete, and downstream jobs reuse the saved artifacts.
If only GitHub release publication fails after PyPI succeeds, rerun that failed
job; do not repeat the package upload. Likewise, if only documentation deployment
fails, fix the Pages configuration and rerun the failed jobs. PyPI and GitHub
release publication remain complete.

If an upload stops after publishing some files, inspect the package index and
compare the uploaded files' hashes with `SHA256SUMS` before deciding how to
complete the release. The workflow deliberately fails on existing filenames;
it does not silently skip or replace them. Avoid **Re-run all jobs** after any
upload: rebuilding a published version can produce different files. Changes to
package contents require a new version. Expired artifacts also require a
separate recovery plan, rather than rebuilding under a published version.

### Existing v0.4.1 tag

The archived `v0.4.1` tag retains its earlier workflow, including the retired
`macos-14` runner. GitHub retired that runner on November 2, 2026; affected jobs
stop before compilation, and publication is skipped because it requires every
distribution build to succeed. See the
[runner retirement notice](https://github.com/actions/runner-images/issues/13518).

Rerunning that tag still uses the old runner, even after the fix reaches `main`.
Preserve the archived tag and prepare a new patch release containing the updated
workflow. To check the fixed platform builds before tagging, push the changes
and manually run **Build Distributions** on `main`; this does not publish any
packages. The next release tag uses the single-build procedure above.

## Documentation hosting

The documentation is hosted at
[https://kinepet.salodev.no/](https://kinepet.salodev.no/).
The release workflow builds documentation from the exact release commit and
saves the validated Pages artifact for 90 days. After PyPI and GitHub release
publication succeed, it calls `.github/workflows/deploy-pages.yml` to deploy that
artifact. Failed or unapproved package releases do not update the website.
Relevant pushes to `main` and pull requests validate documentation only.
The site represents the most recently deployed release; it does not keep
separate documentation sites for each version.

**Publish Documentation** (`.github/workflows/pages.yml`) provides manual
publication for redeployment or documentation-only corrections between releases.

### Initial GitHub Pages setup

1. Open the repository's **Settings → Pages**. Under **Build and deployment**,
   set **Source** to **GitHub Actions**. The repository already supplies the
   workflow, so no generated workflow template is needed.
2. Set **Custom domain** to `kinepet.salodev.no` and save it before changing DNS.
   The domain is configured in GitHub's settings; a `CNAME` file in the repository
   is not required for this Actions-based deployment.
3. In **Settings → Environments**, create or open `github-pages`. Under
   **Deployment branches and tags**, choose **Selected branches and tags** and
   add a **Tag** rule matching `v*` for releases and a **Branch** rule for `main`
   for manual deployments. Leave required reviewers unset for automatic
   documentation deployment after release approval. The `pypi` and `testpypi` environments are independent.
4. Push the committed workflow and documentation changes to `main`. This only
   validates the site. For an initial deployment, manually publish an existing
   release using the procedure below, or wait for a successful package release.
   Keep any current live site until the first release-driven deployment succeeds.
5. In the DNS zone for `salodev.no`, replace the existing record for `kinepet`
   with the CNAME below. Remove conflicting A or AAAA records for that same
   hostname; leave records for the rest of the domain unchanged.

   | Type | Name / host | Target / value | TTL |
   | --- | --- | --- | --- |
   | CNAME | `kinepet` | `c-salomonsen.github.io` | 300 seconds or provider default |

   The target is a hostname: omit `https://` and `/KinePET/`. If the DNS provider
   offers a proxy, use DNS-only mode during setup.
6. Wait for the DNS check and HTTPS certificate provisioning in **Settings →
   Pages**, then enable **Enforce HTTPS** when available. DNS propagation and
   certificate availability can take up to 24 hours. Check the CNAME with:

   ```bash
   dig +short CNAME kinepet.salodev.no
   ```

   It should return `c-salomonsen.github.io.`. Open
   `https://kinepet.salodev.no/` and check the home page, model figure, navigation,
   and a nested API page after deployment. Keep the old
   host serving documentation until the DNS transition is complete.

See [GitHub's custom-domain instructions](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/managing-a-custom-domain-for-your-github-pages-site).

The workflow uses GitHub's built-in token and OIDC permissions. No additional
API token, repository secret, or `gh-pages` branch is required. See
[GitHub's Pages workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

### Manual publication and documentation corrections

Select **Actions → Publish Documentation → Run workflow**, choose `main` for the
workflow version, and supply the source release tag, branch, or commit to build.
To redeploy `v0.4.2` after that release exists:

```bash
gh workflow run pages.yml --repo c-salomonsen/KinePET --ref main -f source_ref=v0.4.2
```

The workflow definition comes from `main`, but checkout and the documentation
build use `source_ref`. There is no default source. Choose a released tag for
ordinary redeployment. Running this manual workflow from a branch other than
`main` skips publication. The separate **Documentation** workflow performs
validation only.

For corrections between releases, create a branch from the currently published
release tag, apply only the necessary documentation changes, and push that
branch. Then use its branch name or, preferably, exact commit as `source_ref`.
Review the diff against the release tag before publication; avoid using ongoing
`main` development as the source. Merge the correction into normal development
as well so the next release retains it. Manual publication replaces the live
site and does not publish a package or move the release tag.

A future URL change requires coordinated updates to `site_url` in `mkdocs.yml`,
package project URLs, README and citation links, and any old-host redirects.
The custom domain serves documentation at `/`; do not add a `/KinePET/` prefix
to `site_url` or public documentation links.
