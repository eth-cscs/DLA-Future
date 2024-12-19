# Release procedure for DLA-Future

DLA-Future follows [Semantic Versioning](https://semver.org).

1. For minor and major releases: check out the `master` branch. All changes required for the release are
   added to `master` via pull requests. For patch releases: check out the corresponding
   `version_major.minor` branch.

1. Write release notes in `CHANGELOG.md`. Check for issues and pull requests for the release on the
   [DLA-F Planning board](https://github.com/orgs/eth-cscs/projects/1). Make sure to include changes that
   may affect users, such as API changes, bugfixes, performance improvements, dependency updates. Changes
   that do not directly affect users may be left out, such as CI changes, miscellaneous spack package
   updates, documentation updates, or utility script updates. All list entries and paragraphs must be on
   a single line for correct formatting on GitHub.

1. Update the version in `CMakeLists.txt`.

1. Update the version and date in `CITATION.cff`.

1. When making a post-1.0.0 major release, remove deprecated functionality if
   appropriate.

1. Update the minimum required versions if necessary.

1. Add a link to the documentation for the release in `DOCUMENTATION.md` and update the link in `README.md`.
   The documentation will be generated automatically after the `vx.y.z` tag has been created and pushed.

1. Ensure you have `gh` ([GitHub CLI](https://cli.github.com)) installed. Run `gh auth login` to authenticate
   with your GitHub account, or set `GITHUB_TOKEN` to a token with `public_repo` access.

1. Create a release on GitHub using the script `scripts/roll_release.sh`. This
   script automatically tags the release with the corresponding release number.

1. Update spack recipe in `spack/packages/dla-future/package.py` adding the new release.

1. Synchronize [upstream spack
   package](https://github.com/spack/spack/blob/develop/var/spack/repos/builtin/packages/dla-future/package.py)
   with local repository. Exclude blocks delimited by `###` comments. These are only intended for the
   internal spack package.

1. For patch releases: update `CHANGELOG` of `master` branch.

1. Delete your `GITHUB_TOKEN` if created only for the release.

1. Modify the release procedure if necessary.
