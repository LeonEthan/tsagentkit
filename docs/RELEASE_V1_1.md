# v1.1 Release Checklist and Migration Note

This note is the Phase 7 release gate for `tsagentkit` v1.1.

## Release Checklist (Must Pass Before Tag)

Use this checklist before creating any `v1.1.x` tag:

- [ ] Stable API contract reviewed against `docs/API_STABILITY.md` with no unannounced breaking signature changes.
- [ ] `TaskSpec` compatibility aliases remain valid (`horizon -> h`, `require_tsfm -> tsfm_policy.mode`, `tsfm_preference -> tsfm_policy.adapters`).
- [ ] TSFM default policy confirmed as required:
  - `TSFMPolicy.mode == "required"` by default.
  - Non-TSFM fallback requires explicit opt-in (`mode="preferred"` or `mode="disabled"`).
- [ ] Error-code behavior confirmed for TSFM hard requirement:
  - Unavailable TSFM under required policy raises `E_TSFM_REQUIRED_UNAVAILABLE`.
- [ ] Artifact lifecycle compatibility checks are green:
  - `save_run_artifact()`
  - `load_run_artifact()`
  - `validate_run_artifact_for_serving()`
  - `replay_forecast_from_artifact()`
- [ ] CI release gates are green on the release commit:
  - `test` (Python matrix)
  - `tsfm_policy_matrix` (deterministic policy cells)
  - `real_tsfm_smoke` (non-mock minimal adapter smoke)
- [ ] Build gate depends on all required checks:
  - `build.needs: [test, tsfm_policy_matrix, real_tsfm_smoke]`
- [ ] Docs are consistent with assembly-first and TSFM-required posture:
  - `README.md`
  - `docs/API_STABILITY.md`
  - `docs/ARCHITECTURE.md`
  - `skill/*.md` and `src/tsagentkit/skill/*.md`

## Reference Validation Commands

```bash
uv run pytest
uv run pytest -v tests/ci/test_tsfm_policy_matrix.py
uv run pytest -v tests/ci/test_real_tsfm_smoke_gate.py
TSFM_RUN_REAL=1 uv run pytest -v tests/ci/test_real_tsfm_smoke_gate.py
uv run pytest tests/docs/test_readme_examples.py tests/docs/test_skill_examples.py tests/docs/test_phase6_doc_consistency.py
```

## Migration Note (v1.0 -> v1.1)

### 1. Default TSFM Policy Is Now Strict

Behavioral change:
- `TaskSpec` now defaults to `tsfm_policy.mode="required"`.
- If no TSFM adapter is available, planning fails fast with `E_TSFM_REQUIRED_UNAVAILABLE`.

Why:
- Keep TSFM as first-class production path for coding-agent-built forecasting systems.
- Prevent silent downgrade to non-TSFM models in production.

### 2. How to Keep Legacy Fallback Behavior

If you intentionally want legacy relaxed behavior, set policy explicitly:

```python
from tsagentkit import TaskSpec

# Legacy-like behavior: prefer TSFM but allow non-TSFM fallback
spec = TaskSpec(
    h=14,
    freq="D",
    tsfm_policy={"mode": "preferred"},
)
```

Or explicitly disable TSFM routing:

```python
spec = TaskSpec(
    h=14,
    freq="D",
    tsfm_policy={"mode": "disabled"},
)
```

### 3. CI and Release Gate Changes

Release eligibility now includes two TSFM-specific gates:

- Deterministic policy matrix checks (`tests/ci/test_tsfm_policy_matrix.py`)
- Real non-mock adapter smoke gate (`tests/ci/test_real_tsfm_smoke_gate.py` with `TSFM_RUN_REAL=1`)

The PyPI build job is blocked unless all required jobs are green.

### 4. Upgrade Actions for Integrators

1. Audit all `TaskSpec(...)` call sites that omit `tsfm_policy`.
2. For production TSFM posture, keep default `required` and ensure at least one adapter is available.
3. For controlled non-TSFM fallback, set `tsfm_policy.mode="preferred"` explicitly.
4. Update operational alerting to recognize `E_TSFM_REQUIRED_UNAVAILABLE`.
5. Add the real smoke command to pre-release validation:
   `TSFM_RUN_REAL=1 uv run pytest -v tests/ci/test_real_tsfm_smoke_gate.py`.
