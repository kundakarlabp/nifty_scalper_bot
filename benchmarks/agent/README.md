# Historical regression benchmark

This benchmark is an offline coding-agent quality set built from real repository regression areas.

It answers a narrow question: **does a proposed engineering change still satisfy the historical invariants that previously caused expensive or safety-relevant failures?**

Usage:

```bash
python scripts/agent_benchmark.py --validate
python scripts/agent_benchmark.py --list
python scripts/agent_benchmark.py --run
```

The manifest deliberately references canonical regression suites rather than copying their logic. Update a case only when the owning regression test or invariant changes.

This benchmark is not a profitability benchmark and does not replace full CI.
