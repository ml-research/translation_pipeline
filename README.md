# benchmark-translation

Translate HuggingFace datasets to multiple languages using vLLM.

## Python and uv

- Requires Python >= 3.12.
- This project expects uv to install Python in a shared location:
  - `UV_PYTHON_INSTALL_DIR=/mnt/vast/workspaces/jackal_ai/.uv/python`

If you use direnv, `.envrc` is provided. Otherwise, export the variable manually:

```bash
export UV_PYTHON_INSTALL_DIR=/mnt/vast/workspaces/jackal_ai/.uv/python
```
