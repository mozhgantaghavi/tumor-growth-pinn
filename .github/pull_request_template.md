## Summary
What does this change and why?

## Checklist
- [ ] `ruff check src tests scripts examples` passes
- [ ] `pytest` passes
- [ ] `node --test "tests/js/*.test.mjs"` passes
- [ ] If the numerical scheme changed, `docs/js/solver.js` matches `src/tumorpinn/solver.py` and `python -m tumorpinn export` was re-run
- [ ] New data sources are public, licensed for redistribution, and attributed in `data/real/README.md`
- [ ] Figures contain no arrows (project convention)
