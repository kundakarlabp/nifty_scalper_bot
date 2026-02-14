from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import subprocess
import os

app = FastAPI()

ENV_PATH = "/home/ubuntu/nifty_scalper_bot/.env"
PASSWORD = "CHANGE_THIS_PASSWORD"

def update_env(key, value):
    lines = []
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r") as f:
            lines = f.readlines()

    updated = False
    for i, line in enumerate(lines):
        if line.startswith(key + "="):
            lines[i] = f"{key}={value}\n"
            updated = True

    if not updated:
        lines.append(f"{key}={value}\n")

    with open(ENV_PATH, "w") as f:
        f.writelines(lines)


def get_service_status(service):
    result = subprocess.run(
        ["systemctl", "is-active", service],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()


def get_memory_usage():
    result = subprocess.run(["free", "-h"], capture_output=True, text=True)
    return result.stdout


def get_logs():
    result = subprocess.run(
        ["journalctl", "-u", "nifty-scalper", "-n", "50", "--no-pager"],
        capture_output=True,
        text=True
    )
    return result.stdout


@app.get("/", response_class=HTMLResponse)
def dashboard():

    bot_status = get_service_status("nifty-scalper")
    admin_status = get_service_status("nifty-admin")
    memory = get_memory_usage()

    bot_color = "success" if bot_status == "active" else "danger"

    return f"""
    <html>
    <head>
        <title>Nifty Scalper Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-dark text-white">

    <div class="container mt-4">

        <h2 class="mb-4">📊 Nifty Scalper Dashboard</h2>

        <div class="row">
            <div class="col-md-6">
                <div class="card bg-secondary">
                    <div class="card-body">
                        <h5>Bot Status</h5>
                        <span class="badge bg-{bot_color}">{bot_status.upper()}</span>
                    </div>
                </div>
            </div>

            <div class="col-md-6">
                <div class="card bg-secondary">
                    <div class="card-body">
                        <h5>Admin Service</h5>
                        <span class="badge bg-success">{admin_status.upper()}</span>
                    </div>
                </div>
            </div>
        </div>

        <br>

        <div class="card bg-secondary">
            <div class="card-body">
                <h5>Memory Usage</h5>
                <pre>{memory}</pre>
            </div>
        </div>

        <br>

        <div class="card bg-secondary">
            <div class="card-body">
                <h5>Control Panel</h5>
                <form method="post" action="/update">
                    Password:<br>
                    <input type="password" name="password" class="form-control"><br>

                    Kite API Key:<br>
                    <input type="text" name="api_key" class="form-control"><br>

                    Access Token:<br>
                    <input type="text" name="access_token" class="form-control"><br>

                    Mode:<br>
                    <select name="mode" class="form-control">
                        <option value="SHADOW">SHADOW</option>
                        <option value="LIVE">LIVE</option>
                    </select><br>

                    <button type="submit" class="btn btn-warning">Save & Restart</button>
                </form>

                <br>
                <a href="/restart" class="btn btn-danger">Restart Bot</a>
                <a href="/logs" class="btn btn-info">View Logs</a>

            </div>
        </div>

    </div>
    </body>
    </html>
    """


@app.post("/update")
def update(password: str = Form(...),
           api_key: str = Form(...),
           access_token: str = Form(...),
           mode: str = Form(...)):

    if password != PASSWORD:
        return {"error": "Wrong password"}

    update_env("KITE_API_KEY", api_key)
    update_env("KITE_ACCESS_TOKEN", access_token)
    update_env("EXECUTION_MODE", mode)
    update_env("ENABLE_LIVE", "true" if mode == "LIVE" else "false")

    subprocess.run(["systemctl", "restart", "nifty-scalper"])

    return {"status": "Updated & Restarted"}


@app.get("/restart")
def restart():
    subprocess.run(["systemctl", "restart", "nifty-scalper"])
    return {"status": "Bot Restarted"}


@app.get("/logs", response_class=HTMLResponse)
def logs():
    logs = get_logs()
    return f"<pre>{logs}</pre>"
