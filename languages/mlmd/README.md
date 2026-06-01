# MLMD Language Config

This directory contains Zed language configuration files for the MLMD DSL:
`config.toml`, `highlights.scm`, `brackets.scm`, `indents.scm`, `outline.scm`,
and `language-configuration.json`.

These are bundled in the Zed dev extension (`extension.toml` at the project root
references `languages/mlmd/`).

## Installation

From the repo root:

```bash
mlmd install zed
```

This sets up the full dev extension at the project root, including these
language config files. No manual copying needed.
