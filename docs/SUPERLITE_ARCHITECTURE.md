# Lightweight UI isolation

The admin controls and read-only review page run as separate processes and separate bounded services from the trading engine. Both use the external instance environment without overwriting unspecified values. The review service remains read-only; the admin service handles daily token and operational controls.
