# Execution safety validation failure

Failed stage: full-tests
Exit status: 1

## patch
```text
```

## compile
```text
```

## focused
```text
.............................................                            [100%]
```

## full
```text

==================================== ERRORS ====================================
____________ ERROR collecting tests/dashboard/test_console_smoke.py ____________
ImportError while importing test module '/home/runner/work/nifty_scalper_bot/nifty_scalper_bot/tests/dashboard/test_console_smoke.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
tests/dashboard/test_console_smoke.py:5: in <module>
    from streamlit.testing.v1 import AppTest
E   ModuleNotFoundError: No module named 'streamlit'
=========================== short test summary info ============================
ERROR tests/dashboard/test_console_smoke.py
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!
```
