# Maintainers Guide

This page contains instructions for project maintainers about how our setup works,
making releases, creating packages, etc.

If you want to make a contribution to the project, see the
[Contributing Guide](CONTRIBUTING.md) instead.


## Branches

* *master*: Always tested and ready to become a new version. Don't push directly to this
  branch. Make a new branch and submit a pull request instead.
* *gh-pages*: Holds the HTML documentation and is served by Github. Pages for the master
  branch are in the `dev` folder. Pages for each release are in their own folders.
  **Automatically updated by TravisCI** so you shouldn't have to make commits here.


## Reviewing and merging pull requests

A few guidelines for reviewing:

* Always **be polite** and give constructive feedback.
* Welcome new users and thank them for their time, even if we don't plan on merging the
  PR.
* Don't be harsh with code style or performance. If the code is bad, either (1) merge
  the pull request and open a new one fixing the code and pinging the original submitter
  (2) comment on the PR detailing how the code could be improved. Both ways are focused
  on showing the contributor **how to write good code**, not shaming them.

Pull requests should be **squash merged**.
This means that all commits will be collapsed into one.
The main advantages of this are:

* Eliminates experimental commits or commits to undo previous changes.
* Makes sure every commit on master passes the tests and has a defined purpose.
* The maintainer writes the final commit message, so we can make sure it's good and
  descriptive.


## Continuous Integration

We use TravisCI and AppVeyor continuous integration (CI) services to build and test the
project on Windows, Linux, and Mac.
The configuration files for these services are `.travis.yml` and `.appveyor.yml`.
Both rely on the `requirements.txt` file to install the required dependencies using
conda and the `Makefile` to run the tests and checks.

Travis also handles all of our deployments automatically:

* Updating the development documentation by pushing the built HTML pages from the
  *master* branch onto the `dev` folder of the *gh-pages* branch.
* Uploading new releases to PyPI (only when the build was triggered by a git tag).
* Updated the `latest` documentation link to the new release.

This way, most day-to-day maintenance operations are automatic.

The scripts that setup the test environment and run the deployments are loaded from the
[fatiando/continuous-integration](https://github.com/fatiando/continuous-integration)
repository to avoid duplicating work across multiple repositories.
If you find any problems with the test setup and deployment, please create issues and
submit pull requests to that repository.


## Making a Release

We try to automate the release process as much as possible.
Travis handles publishing new releases to PyPI and updating the documentation.
The version number is set automatically using versioneer based information it gets from
git.
There are a few steps that still must be done manually, though.

### Drafting a new Zenodo release

If the project already has releases on [Zenodo](https://zenodo.org/), you need to create
a **New version** of it. Find the link to the latest Zenodo release on the `README.md`
file of your project. Then:

1. Delete all existing files (they will be replaced with the new version).
2. Reserve a DOI and save the release draft.
3. Include as authors anyone who made contributions between now and the last release.

On the other hand, if you're making the first release of the project, you need to create
a **New upload** for it inside the
[Fatiando a Terra community at Zenodo](https://zenodo.org/communities/fatiando/).

1. Make sure the Fatiando a Terra community is chosen when filling the release draft.
2. Reserve a DOI and save the release draft.
3. Include as authors anyone who made contributions between now and the last release.

### Updating the changelog

1. Generate a list of commits between the last release tag and now:

    ```bash
    git log HEAD...v0.1.2 > changes.txt
    ```

2. Edit the changes list to remove any trivial changes (updates to the README, typo
   fixes, CI configuration, etc).
3. Replace the PR number in the commit titles with a link to the Github PR page. In Vim,
   use `` %s$#\([0-9]\+\)$`#\1 <https://github.com/fatiando/PROJECT/pull/\1>`__$g ``
   to make the change automatically.
4. Copy the remaining changes to `doc/changes.rst` under a new section for the
   intended release.
5. Include the DOI badge in the changelog. Remember to replace your DOI inside the badge
   url.

    ```
    .. image:: https://img.shields.io/badge/doi-<DOI-FIRST-PART>%2F<DOI-SECOND-PART>-blue.svg?style=flat-square
        :alt: Digital Object Identifier for the Zenodo archive
        :target: https://doi.org/<INSERT-YOUR-DOI>
    ```

6. Add a link to the new release version documentation in `README.rst`.
7. Open a new PR with the updated changelog.

### Check the README syntax

Github is a bit forgiving when it comes to the RST syntax in the README but PyPI is not.
So slightly broken RST can cause the PyPI page to not render the correct content. Check
using the `rst2html.py` script that comes with docutils:

```
python setup.py --long-description | rst2html.py --no-raw > index.html
```

Open `index.html` and check for any flaws or error messages.

### Pushing to PyPI and updating the documentation

After the changelog is updated, making a release should be as simple as creating a new
git tag and pushing it to Github:

```bash
git tag v0.2.0
git push --tags
```

The tag should be version number (following [Semantic Versioning](https://semver.org/))
with a leading `v`.
This should trigger Travis to do all the work for us.
A new source distribution will be uploaded to PyPI, a new folder with the documentation
HTML will be pushed to *gh-pages*, and the `latest` link will be updated to point to
this new folder.

### Archiving on Zenodo

Grab a zip file from the Github release and upload to Zenodo using the previously
reserved DOI.

### Updating the conda package

After Travis is done building the tag and all builds pass, we need to update the conda
package.
Unfortunately, this needs to be done manually for now.

1. Fork the feedstock repository (https://github.com/conda-forge/PROJECT-feedstock) if
   you haven't already. If you have a fork, update it.
2. Update the version number and sha256 hash on `recipe/meta.yaml`. You can get the hash
   from the PyPI "Download files" section.
3. Add or remove any new dependencies (most are probably only `run` dependencies).
4. Make a new branch, commit, and push your changes **to your fork**.
5. Create a PR against the original feedstock master.
6. Once the CIs are passing, merge or as a maintainer to do so.
