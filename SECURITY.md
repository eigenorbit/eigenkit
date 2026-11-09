# Security policy

## Supported versions

Security fixes go into the latest release of `eigenkit`. Older versions are
not patched.

## Reporting a vulnerability

Please do not report a security vulnerability in a public GitHub issue.

Report it privately instead, either through GitHub's
[private vulnerability reporting](https://github.com/eigenorbit/eigenkit/security/advisories/new)
or by e-mail to **info@eigenorbit.com**. Include as much of the following as
you can:

- what the issue is and what it lets an attacker do;
- the steps to reproduce it, ideally a minimal code sample;
- the `eigenkit` and Python versions you used.

We will acknowledge the report, keep you informed while we work on a fix, and
credit you in the release notes unless you prefer to stay anonymous. Please
keep the issue private until a fixed release is out.

Credentials deserve a note: eigenkit never writes provider tokens (IBM, IQM)
into saved models, pickles or `repr()` output, and a saved model cannot choose
the host its provider connects to. If you find a path that does either, that
is a security issue and belongs in a private report.

Saved models (`.eigenkit` bundles) hold no pickles, and loading one does not
run code from it. If a bundle runs code when you load it without
`trusted=True`, that is a security issue and belongs in a private report.
Bundles saved before eigenkit 1.3 hold joblib pickles and load only with
`trusted=True`. Code execution from a bundle loaded that way is expected
behaviour, as with `torch.load` or `joblib.load`, not a vulnerability.
