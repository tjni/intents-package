# Home Assistant Intents Package

Packaging for [intents](https://github.com/OHF-Voice/intents/)


## Install

Clone the repo and create a virtual environment:

``` sh
git clone --recursive https://github.com/OHF-Voice/intents-package
cd intents-package
script/setup
```


## Release

Releases are automated by the
[Release workflow](.github/workflows/publish.yml): pushing a `YYYY.M.D` tag
(no leading `v`) publishes the package to PyPI via [Trusted Publishing][oidc]
and creates the matching GitHub release.

To cut a release:

1. Bump `version` in `pyproject.toml` to `YYYY.M.D` and commit.
2. (Optional) Add a `## YYYY.M.D` section to `CHANGELOG.md`. If present, it
   becomes the release notes; otherwise GitHub auto-generates them.
3. Tag the commit and push the tag:

   ``` sh
   git tag YYYY.M.D
   git push origin YYYY.M.D
   ```

The workflow then:

1. Verifies the tag matches `version` in `pyproject.toml`.
2. Checks out the `intents` submodule at the **same tag** and commits the
   bumped pointer.
3. Runs `script/package` (build + sanity check).
4. Only after that passes, pushes the submodule bump to the default branch and
   moves the tag onto that commit, so the tag, the default branch, and the
   published artifacts all point at the same commit.
5. Publishes to PyPI and creates the GitHub release.

The `intents` repo must already have a tag with the matching `YYYY.M.D` name,
and the version must match `pyproject.toml`, or the workflow fails before
publishing anything.

[oidc]: https://docs.pypi.org/trusted-publishers/
