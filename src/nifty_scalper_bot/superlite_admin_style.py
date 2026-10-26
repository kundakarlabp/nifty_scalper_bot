"""Single source of truth for the mobile-first admin CSS and progressive enhancement."""

# Embedded HTML/CSS/JS assets have intentional long literal lines.
# ruff: noqa: E501

# fmt: off  # CSS and JavaScript are intentionally authored in their native syntax.
STYLE = r"""
:root{color-scheme:dark;--bg:#080d15;--panel:#111c2a;--panel2:#142335;--line:#293b51;--text:#edf4fb;--mut:#a9b9cb;--green:#38d6a0;--amber:#f5bb5a;--red:#ff7586;--blue:#70b4ff;--focus:#a8d8ff}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:96px}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--blue)}.top{position:sticky;top:0;z-index:10;padding:10px max(16px,calc((100vw - 1112px)/2));background:#0b1522f7;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:12px;flex-wrap:wrap;backdrop-filter:blur(14px)}
.brand{display:flex;align-items:center;gap:9px;min-width:0;font-size:16px;font-weight:800;color:var(--text);text-decoration:none}.logo{display:grid;place-items:center;width:34px;height:34px;border-radius:10px;background:#153451;color:var(--blue)}.sp{flex:1}.nav{display:flex;gap:4px;align-items:center;flex-wrap:wrap}.nav a{padding:8px 9px;border-radius:8px;color:var(--mut);text-decoration:none;font-size:13px}.nav a:hover{background:var(--panel2);color:var(--text)}
.wrap{max-width:1120px;margin:0 auto;padding:21px 16px 72px}.intro{display:flex;align-items:end;justify-content:space-between;gap:10px;flex-wrap:wrap;margin-bottom:15px}.intro h1{margin:0;font-size:clamp(21px,3vw,29px);letter-spacing:-.04em}.eyebrow{font-size:11px;letter-spacing:.15em;text-transform:uppercase;color:var(--mut);font-weight:750}.intro p{margin:1px 0 0}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:13px}.layout{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(300px,1fr);align-items:start;gap:13px}.stack{min-width:0}.card{background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:19px;margin-bottom:13px;box-shadow:0 5px 19px #0002}.card h2{font-size:17px;letter-spacing:-.02em;margin:0 0 5px}.card p{margin:4px 0 12px;color:var(--mut);font-size:13px}.metric{background:#101b28;border:1px solid var(--line);border-radius:12px;padding:13px;min-height:91px}.metric .k{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em}.metric .v{font-size:17px;font-weight:800;line-height:1.45;overflow-wrap:anywhere;margin-top:3px}.metric small{display:block;overflow-wrap:anywhere;font-size:11px;color:var(--mut)}
.pill{display:inline-flex;align-items:center;gap:5px;padding:5px 10px;font-size:11px;border:1px solid var(--line);border-radius:999px;letter-spacing:.04em;font-weight:800}.pill.on{color:var(--red);background:#401c2a;border-color:#784156}.pill.off{color:var(--green);background:#103127;border-color:#24654a}.pill.unknown{color:var(--amber)}.good{color:var(--green)}.warning{color:var(--amber)}.failure{color:var(--red)}.muted{color:var(--mut);font-size:12px}.hint{color:var(--mut);font-size:12px;margin-top:10px}.subheading{font-size:11px;letter-spacing:.11em;font-weight:800;text-transform:uppercase;color:var(--mut);margin:14px 0 6px}
label{display:block;font-size:12px;color:var(--mut);font-weight:600;margin:12px 0 5px}input,select{width:100%;min-height:45px;padding:11px 12px;border-radius:9px;border:1px solid #395067;background:#0b1523;color:var(--text);font:inherit;font-size:15px}input::placeholder{color:#7d91a7}input:focus-visible,button:focus-visible,a:focus-visible,summary:focus-visible,select:focus-visible{outline:3px solid var(--focus);outline-offset:2px}
button,.btn{display:inline-flex;align-items:center;justify-content:center;gap:7px;min-height:44px;max-width:100%;padding:11px 16px;border:1px solid transparent;border-radius:10px;background:#267bd6;color:#fff;font:700 14px system-ui,sans-serif;cursor:pointer;text-decoration:none;white-space:normal;text-align:center}button:hover,.btn:hover{filter:brightness(1.12)}button:disabled{opacity:.58;cursor:wait}button.amb,.btn.amb,button.amber{background:#80510f;color:#fff}button.green,.btn.green{background:#11785b}button.gray,.btn.gray{background:#26384b;border-color:#395067}button.danger{background:#783244}.actions{display:flex;gap:9px;flex-wrap:wrap;align-items:center}.actions>*{flex:0 1 auto}.actions form{margin:0}.actions .btn,.actions button{margin:0}.full{width:100%}.flow{display:grid;gap:11px}.flow form{margin:0}.session{border-color:#33698d;background:linear-gradient(145deg,#162b40,#111c2a)}.session .primary{background:#237ad6;font-size:16px;min-height:51px}.steps{margin:8px 0 12px;padding-left:19px;color:var(--mut);font-size:13px}.steps li{margin:5px 0}.badge{display:inline-block;background:#182739;border:1px solid var(--line);border-radius:6px;padding:2px 6px;color:var(--mut);font-size:10px}
details{margin-top:10px;border-top:1px solid var(--line)}summary{cursor:pointer;min-height:46px;display:flex;align-items:center;justify-content:space-between;font-size:14px;font-weight:750;list-style:none;padding:12px 2px}summary::-webkit-details-marker{display:none}summary:after{content:"＋";color:var(--blue)}details[open]>summary:after{content:"−"}.detail-body{padding:0 2px 9px}.flash{padding:13px 14px;border-radius:10px;margin:0 0 16px;border:1px solid #33516b;background:#123049;color:#c8e6ff}.flash.ok{border-color:#24654a;background:#103127;color:#a7f5d8}.flash.err{border-color:#784156;background:#401c2a;color:#ffd1d8}
.console{margin:11px 0 0;background:#060e17;color:#d4e2f0;border:1px solid #32465e;border-radius:10px;padding:12px;max-height:320px;min-height:135px;overflow:auto;font:12px/1.6 ui-monospace,"SFMono-Regular",Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere}.statusline{display:flex;align-items:center;gap:9px;flex-wrap:wrap;font-size:12px;color:var(--mut)}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--amber)}.dot.good{background:var(--green)}.dot.bad{background:var(--red)}.right{margin-left:auto}.split{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.cta-strip{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;margin:7px 0 14px}.sidehelp{border-left:3px solid #365e7e;padding:5px 11px;color:var(--mut);font-size:12px}.footer{color:var(--mut);font-size:11px;margin-top:18px}
@media(max-width:800px){.layout{grid-template-columns:1fr}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.top{padding:9px 14px}.wrap{padding:17px 13px 80px}.nav a{padding:7px}.card{padding:16px}.intro{margin-bottom:12px}}@media(max-width:450px){.brand{font-size:14px}.logo{width:31px;height:31px}.top{gap:6px}.top .pill{margin-left:auto}.nav{order:5;width:100%;justify-content:space-between}.nav a{flex:1;text-align:center;font-size:12px}.metric{padding:11px;min-height:83px}.metric .v{font-size:15px}.actions{align-items:stretch}.actions form{flex:1 1 100%}.actions button,.actions .btn{width:100%}.split{grid-template-columns:1fr}.wrap{padding:14px 10px 65px}.card{padding:14px}.intro h1{font-size:22px}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
"""

SCRIPT = r"""
(function () {
  "use strict";
  var endpoint = "/admin/api/overview";
  var status = document.getElementById("status-note");
  var refresh = document.getElementById("refresh-status");
  var running = false;
  var lastSuccess = 0;

  function setText(id, value) {
    var node = document.getElementById(id);
    if (node) node.textContent = String(value == null || value === "" ? "—" : value);
  }
  function highlight(id, state) {
    var node = document.getElementById(id);
    if (node) node.className = "v " + state;
  }
  function labelBroker(value) {
    var word = String(value || "unknown").toLowerCase();
    if (word === "true" || word === "authenticated" || word === "yes") return "Authenticated";
    if (word === "false" || word === "unauthenticated" || word === "no") return "Login required";
    return "Unknown";
  }
  function getJSON(url, timeoutMs) {
    var controller = new AbortController();
    var timeout = setTimeout(function () { controller.abort(); }, timeoutMs);
    return fetch(url, { credentials:"same-origin", cache:"no-store", signal:controller.signal })
      .then(function (response) {
        if (!response.ok) throw Error("HTTP " + response.status);
        return response.json();
      }).finally(function () { clearTimeout(timeout); });
  }
  function drawOverview(data) {
    var online = data.engine_ok === true;
    var quiet = online && data.quiet === true;
    setText("engine-value", !online ? "Unreachable" : quiet ? "QUIET (planned)" : data.bot_loaded ? "Running" : "Starting");
    highlight("engine-value", !online ? "failure" : quiet || data.bot_loaded ? "good" : "warning");
    setText("broker-value", online && !quiet ? labelBroker(data.broker_authenticated) : "Not checked");
    highlight("broker-value", online && !quiet && labelBroker(data.broker_authenticated) === "Authenticated" ? "good" : "warning");
    setText("mode-value", quiet ? "AUTO · QUIET" : online ? data.mode : "Unknown");
    setText("blocker-value", quiet ? "Scheduled idle" : online ? (data.primary_blocker || "None") : "Engine unavailable");
    setText("release-value", data.updater_state || "Unknown");
    setText("revision-value", online && data.runtime_sha ? data.runtime_sha.slice(0, 10) : "—");
    var result = !online ? "Engine unavailable — check service" : quiet ? "AUTO quiet · control plane online" : "Engine reachable";
    if (data.updater_state && /deploying|restarting|validating|recovering|fetching/.test(data.updater_state))
      result += " · Deployment in progress";
    status.textContent = result + " · " + new Date().toLocaleTimeString();
    status.className = online ? "good" : "failure";
    lastSuccess = Date.now();
  }
  function poll(force) {
    if (running || (!force && document.hidden)) return;
    running = true;
    status.textContent = "Checking live status…";
    status.className = "";
    getJSON(endpoint, 6000).then(drawOverview).catch(function () {
      status.textContent = "Status unavailable · last successful check " +
        (lastSuccess ? new Date(lastSuccess).toLocaleTimeString() : "never");
      status.className = "failure";
      setText("engine-value", "Unknown");
      highlight("engine-value", "warning");
    }).finally(function () { running = false; });
  }
  if (status) {
    poll(true);
    window.setInterval(function () { poll(false); }, 20000);
    document.addEventListener("visibilitychange", function () { if (!document.hidden) poll(true); });
    if (refresh) refresh.addEventListener("click", function () { poll(true); });
  }

  var logOutput = document.getElementById("log-output");
  var logButton = document.getElementById("load-logs");
  var logFilter = document.getElementById("log-filter");
  var logQuery = document.getElementById("log-query");
  var logLines = document.getElementById("log-lines");
  function logs() {
    if (!logOutput || !logButton) return;
    logButton.disabled = true;
    logOutput.textContent = "Loading bounded logs…";
    var query = logFilter.value === "custom" ? logQuery.value.trim() : logFilter.value;
    var params = new URLSearchParams({ lines: logLines.value, contains: query });
    getJSON("/admin/api/logs?" + params.toString(), 14000).then(function (data) {
      logOutput.textContent = data.text || "No matching log lines.";
    }).catch(function (error) {
      var denied = /HTTP (401|403)/.test(String(error && error.message || error));
      logOutput.textContent = denied
        ? "Detailed logs require this browser to be verified using the existing daily Zerodha token field above. The engine and broker may already be healthy; see live status above."
        : "Live logs could not be loaded. Check admin service health or retry.";
    }).finally(function () { logButton.disabled = false; });
  }
  if (logButton) {
    logButton.addEventListener("click", logs);
    logFilter.addEventListener("change", function () {
      logQuery.hidden = logFilter.value !== "custom";
      if (logFilter.value !== "custom") logs();
    });
    logQuery.addEventListener("keydown", function (event) { if (event.key === "Enter") logs(); });
  }

  document.querySelectorAll("form").forEach(function (form) {
    form.addEventListener("submit", function () {
      var button = form.querySelector('button[type="submit"]');
      if (!button || button.disabled) return;
      button.disabled = true;
      button.textContent = "Submitting…";
    });
  });
})();
"""

# fmt: on
