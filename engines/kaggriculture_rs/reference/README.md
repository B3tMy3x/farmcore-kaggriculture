# Pinned reference

Source: https://github.com/Kaggle/kaggle-environments
Commit: `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`.

Files are unmodified upstream sources; see `UPSTREAM.json` for paths and
`LICENSE` for Apache-2.0 terms. `core.py` and `utils.py` document framework
step numbering, reward handling and seed resolution. They are not runtime
dependencies of the Rust package.

`tests/oracle.py` executes the upstream interpreter and seed resolver. Its
small framework adapter supplies observations, configuration and step counters;
`tests/framework.py` also executes the pinned `Environment` class to verify
the adapter against the actual framework. Both exclude agent execution,
wall-clock timeouts and rendering. The simulator
accepts actions supplied by its caller, so remainingOverageTime stays at 60.

`src/constants.json` is generated from the literal tables in this pinned source.
When updating the reference, update the commit, tables and hashes together,
then rerun the full differential suite before accepting the update.
