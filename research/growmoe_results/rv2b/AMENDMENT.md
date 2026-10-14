# RV2-B Pre-Registration Amendment

- **Date and time**: 2026-10-13 10:50 EDT
- **What changed**: `scaling_check.json` was regenerated with `k = 1` (was previously run with default `k = 4` before `bars_rv2b.json` existed). Following the update to `scaling_check.json`, `bars_rv2b.json` (and its SHA256) and `PREREG.md` were regenerated via `python research/growmoe_results/rv2b/run_rv2b.py prereg`. The previous k=4 artifacts (`scaling_check.json`, `bars_rv2b.json`, `bars_rv2b.sha256`, `PREREG.md`) were archived in `_superseded_k4/`.
- **Why**: `dev` selected and `prereg` froze `k = 1`, but `scaling_check.json` had originally been run at `k = 4`. Statement C4 and shipped `gate_shortlist_min_experts` therefore described a k=4 cascade rather than the shipped k=1 cascade. Regenerating scaling at k=1 aligns all statements and parameters with the pre-registered configuration before any confirmatory seed runs.
- **Old and new SHA-256 hashes**:
  - `PREREG.md`:
    - Old: `f80accb8667aaf15b31f6f793f05bc8f6fad655263cb57894df1fd42e70ca6f7`
    - New: `4cb8159a4bbdf3375f90ebd1e8da1130956a8504d1d9d34f046211134df0b624`
  - `bars_rv2b.json`:
    - Old: `af40e922da3bf9e8178a3e339a7007721677729370ab06208f5ddeec8f57fad7`
    - New: `556332a977140d7e06f97c9e46003067e87d369ce5992ed99a2d85ac285ac143`
  - `scaling_check.json`:
    - Old: `134afe666f6b861261559a68a05c2d5c9ca8c1dbbc1d389171a5552baf143c33`
    - New: `81f149a6c0b7a9317c59ac5bff38564c489ed01afa8ac220c20b21557a2e81f7`
  - `dev_k.json`:
    - Old: `34f45aa203548e3c37be75e1e7abb5edeefabb3a7f40e0438d2232cdec1ce26c`
    - New: `34f45aa203548e3c37be75e1e7abb5edeefabb3a7f40e0438d2232cdec1ce26c` (unchanged)
- **Shipped min_experts**:
  - Old: `32`
  - New: `64`
- **C4 ratios at 500 experts (cascade / exact)**:
  - Old (k = 4):
    - B = 64: MACs ratio = 0.019625, Wall-clock time ratio = 0.231381
    - B = 1024: MACs ratio = 0.019625, Wall-clock time ratio = 0.257410
  - New (k = 1):
    - B = 64: MACs ratio = 0.007906, Wall-clock time ratio = 0.214540
    - B = 1024: MACs ratio = 0.007906, Wall-clock time ratio = 0.206908

no file under runs/ existed when this was done
