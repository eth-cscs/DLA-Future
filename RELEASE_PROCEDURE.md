# Release procedure for DLA-Future

DLA-Future follows [Semantic Versioning](https://semver.org).

1. Update the following files via a PR to the `master` branch:

   1. Write release notes in `CHANGELOG.md`. Check for issues and pull requests for the release on the
      [DLA-F Planning board](https://github.com/orgs/eth-cscs/projects/1). Make sure to include changes that
      may affect users, such as API changes, bugfixes, performance improvements, dependency updates. Changes
      that do not directly affect users may be left out, such as CI changes, miscellaneous spack package
      updates, documentation updates, or utility script updates. All list entries and paragraphs must be on
      a single line for correct formatting on GitHub.

   1. Update the version in `CMakeLists.txt`.

   1. Update the versions and date in `CITATION.cff`.

   1. When making a post-1.0.0 major release, remove deprecated functionality if
      appropriate.

   1. Update the minimum required versions if necessary.

   1. Add a link to the documentation for the release in `DOCUMENTATION.md` and update the link in `README.md`.
      The documentation will be generated automatically after the `vX.Y.Z` tag has been created and pushed.

1. Merge all the changes for the release to the `master` branch (also for patch releases), including the above
   changes to version, documentation, etc.

1. Ensure you have `gh` ([GitHub CLI](https://cli.github.com)) installed. Run `gh auth login` to authenticate
   with your GitHub account, or set `GITHUB_TOKEN` to a token with `public_repo` access.

1. Ensure you are working from the `eth-cscs/DLA-Future` and not your own fork for following steps.

1. For minor and major releases: check out the `master` branch. For patch releases: check out the corresponding
   `version_major.minor` branch. If it doesn't exist yet, create it from the `vX.Y.0` tag.

1. For patch releases: cherry pick the commits for the patch from `master` to the `version_X.Y` branch.
   Make sure the cherry-picked commits are listed in the `CHANGELOG.md`, and make sure you cherry-picked
   the PR(s) updating all the above (version, documentation, etc.).

1. Check the release branch to make sure the content matches the release notes.

1. Wait for CI tests to pass before making a release. CI is automatically triggered for `master` and `version_*`
   branches.

1. Create a release on GitHub using the script `scripts/roll_release.sh`. This
   script automatically tags the release with the corresponding release number.

1. Update spack recipe in `spack/packages/dla-future/package.py` adding the new release.

1. Synchronize [upstream spack
   package](https://github.com/spack/spack/blob/develop/var/spack/repos/builtin/packages/dla-future/package.py)
   with local repository. Exclude blocks delimited by `###` comments. These are only intended for the
   internal spack package.

1. Delete your `GITHUB_TOKEN` if created only for the release.

1. Create a new milestone named `vX.Y.Z` for the next major or minor release.

1. Modify the release procedure if necessary.
