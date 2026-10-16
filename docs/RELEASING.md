# Releasing

Steps that can only happen at publication, in order:

1. **Weights.** Create the Hugging Face model repo `m-herre/Comb` (needs a write token), upload `config.json`,
   `model.safetensors` and `hf/README.md` (as `README.md`) from the release weights folder, and tag the commit
   `v<version>`. The two weight files must be byte-identical to the ones `SHA256` in `comb/checkpoint.py` was computed
   from (`weights_digest` of the folder): a download that does not match raises.
2. **Pin.** Set `REVISION` in `comb/checkpoint.py` to that Hugging Face commit hash, so a given package version
   always downloads the same weights.
3. **Date.** Set the release date in `docs/CHANGELOG.md` (replace "unreleased") and add `date-released` to
   `CITATION.cff`.
4. **Check.** `pytest` (CPU), `bench/gpu_check.py` on a GPU, and a clean `pip install` of the built wheel that
   downloads the weights on first `fit`.
5. **Package.** Build and upload `tabcomb` to PyPI; push `main` and the `v<version>` tag to GitHub.
