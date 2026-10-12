# Note on reproducing these pilot scripts

These pilot and spike scripts (2026-10-09) construct `PhyloCfg(...)` directly. Their logged outputs were produced when a newborn expert's competence monitor was armed at birth. Since 2026-10-10 the default is `arm_monitor_after_infancy=True` (see `docs/COMPONENT_CARDS.md`, card 13). To reproduce the logged numbers, add `arm_monitor_after_infancy=False` to the `PhyloCfg(...)` calls.
