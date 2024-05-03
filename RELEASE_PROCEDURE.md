# Release procedure for DLA-Future

DLA-Future follows [Semantic Versioning](https://semver.org).

1. For minor and major releases: create and check out a new branch at an
   appropriate point on `main` with the name `release-major.minor`.  `major` and `minor` should be the
   major and minor versions of the release. For patch releases: check out the corresponding
   `release-major.minor` branch.

1. Write release notes in `CHANGELOG.md`. Check for issues and pull requests for the release on the
   [DLA-F Planning board](https://github.com/orgs/eth-cscs/projects/1). All list entries and paragraphs
   must be on a single line for correct formatting on GitHub.

1. Make sure the version is set correctly in `CMakeLists.txt`.

1. When making a post-1.0.0 major release, remove deprecated functionality if
   appropriate.

1. Update the minimum required versions if necessary.

1. Add a link to the documentation for the release. The documentation will be generated automatically
   after the `vx.y.z` tag has been created and pushed.

1. Create a release on GitHub using the script `scripts/roll_release.sh`. This
   script automatically tags the release with the corresponding release number.  You'll need to set
   `GITHUB_TOKEN` or both `GITHUB_USER` and `GITHUB_PASSWORD` for the hub release command. When creating
   a `GITHUB_TOKEN`, the only access necessary is `public_repo`.

1. Merge release branch into `master` (with `--no-ff`).

1. Modify the release procedure if necessary.

1. Synchronize [upstream spack
   package](https://github.com/spack/spack/blob/develop/var/spack/repos/builtin/packages/dla-future/package.py)
   with local repository.

1. Delete your `GITHUB_TOKEN` if created only for the release.
