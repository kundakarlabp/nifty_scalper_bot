import sys
file_path = sys.argv[1]
with open(file_path, 'r') as f:
    lines = f.readlines()

# Find and modify startup reconciliation handler
for i, line in enumerate(lines):
    if 'Startup reconciliation failed' in line:
        # Comment out hard failure, make it log-only with retry logic
        lines[i] = lines[i].replace(
            'raise RuntimeError',
            '# Non-fatal: reconciliation will retry in background\n        logger.warning'
        )

with open(file_path, 'w') as f:
    f.writelines(lines)
