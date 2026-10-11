# Prompt: fix triplets and add curriculum

The 2606 lane learns provenance and topic, not style. Positives are adjacent rows
(`build_triplets.py:133`, `:481`); validation returns mid-batch (`validate.py:136`)
and reuses train speakers (`import_genshin.py:73`).

Fix the triplet construction, build the curriculum at `automating.md:262`, and
make validation able to see the difference. `FINDINGS.md` has the measurements.

`uv run`, code in `src/`, no GPU run.