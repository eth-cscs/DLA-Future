# Pull Requests management

## Pull Requests title and message

The title of the PR should briefly summarize (one line message) the changes made to the code.
The message should contain a detailed description of the work, including:
- what has been added/changed (classes, methods, functions),
- workarounds (e.g. in case of a library bug),
- any important information contained in commit messages.

## Review

The review consists of checking:
- if the changes match the PR message,
- the documentation (Is it complete?, Is it clear?, ...),
- if tests are complete (Is each public method tested?, do the test cover corner cases?, ...),
- the implementation (code readability, code inefficences, ...),
- Checking if the code is conform to the coding style (not everything can be checked automatically).

## Merging a PR

A PR is ready to merge when all the reviewers (at least one review is required) approve the changes and the CI tests pass.
To avoid a long history of commits which have not been reviewed and tested, the "Squash and Merge" method is used.
When merging the title and message must be completed manually.

The title should reflect the title of the PR and its reference number.
This is the default behaviour in GitHub if the PR contains more than one commit. **Warning**: if the PR contains only one commit, the commit message is used. Please update it.
E.g.
```
scripts: create scripts for all miniapps {strong, weak} x {gpu, mc} + include local benchmarks (#840)
```

The message contains by default a list of the commits and has to be changed.
It can be left empty if the title is self-explanatory, otherwise it should contain extra information. The first PR comment should contain the needed information and can be used (some editing might be required).
E.g.
```
Along gen_dlaf_{strong,weak}, this PR introduces a new pair of generator scripts in order to have a "template" for both {mc,gpu} (specifically it is PizDaint defaults).
Main changes:
- now we have 4 generator scripts for all miniapps gen_dlaf_{strong,weak}-{mc,gpu}
- the strong version of the generator scripts includes an additional "local" benchmark, which runs a set of benchmarks with --local flag; these are useful as reference for distributed single-rank implementations
- all scripts have the main parameters in the beginning, so it is easier to customise the benchmark runs
```
