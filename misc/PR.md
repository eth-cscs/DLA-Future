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
To avoid a long history with commits which do not pass a review the "Squash and Merge" method is used.
The title of the PR (with its reference number) has to be used as commit title, and the PR message as commit message (The default in GitHub).
