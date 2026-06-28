# Execution safety validation failure

Failed stage: followup
Exit status: 1

## patch
```text
Traceback (most recent call last):
  File "/home/runner/work/nifty_scalper_bot/nifty_scalper_bot/tools/_apply_execution_safety_followup.py", line 224, in <module>
    main()
  File "/home/runner/work/nifty_scalper_bot/nifty_scalper_bot/tools/_apply_execution_safety_followup.py", line 218, in main
    patch_position_manager()
  File "/home/runner/work/nifty_scalper_bot/nifty_scalper_bot/tools/_apply_execution_safety_followup.py", line 47, in patch_position_manager
    replace_once(
  File "/home/runner/work/nifty_scalper_bot/nifty_scalper_bot/tools/_apply_execution_safety_followup.py", line 41, in replace_once
    raise RuntimeError(f"{path}: expected one anchor, found {count}: {old[:100]!r}")
RuntimeError: src/nifty_scalper_bot/execution/position_manager.py: expected one anchor, found 0: '        with self._lock:\n            existing_positions = dict(self._positions)\n\n    def _get_float('
```
