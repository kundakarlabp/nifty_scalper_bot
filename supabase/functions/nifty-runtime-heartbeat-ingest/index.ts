const ALLOWED_IP = "65.0.255.84";

function respond(body: Record<string, unknown>, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", "cache-control": "no-store" },
  });
}

function serviceKey(): string {
  try {
    const keys = JSON.parse(Deno.env.get("SUPABASE_SECRET_KEYS") ?? "{}");
    if (typeof keys.default === "string" && keys.default) return keys.default;
  } catch { /* fall back to the platform service role key */ }
  return Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return respond({ ok: false, error: "method_not_allowed" }, 405);
  // Do not trust the client-controlled x-forwarded-for header.
  if (req.headers.get("cf-connecting-ip")?.trim() !== ALLOWED_IP) {
    return respond({ ok: false, error: "source_ip_denied" }, 403);
  }
  if (req.headers.get("x-nifty-source") !== "lightsail-heartbeat-v1") {
    return respond({ ok: false, error: "source_header_denied" }, 403);
  }
  let body: Record<string, unknown>;
  try { body = await req.json(); }
  catch { return respond({ ok: false, error: "invalid_json" }, 400); }

  const stamp = String(body.observed_at ?? "");
  const observedMs = Date.parse(stamp);
  if (!Number.isFinite(observedMs) || Math.abs(Date.now() - observedMs) > 180000) {
    return respond({ ok: false, error: "stale_or_invalid_observation" }, 400);
  }
  const sha = String(body.build_sha ?? "");
  if (sha && !/^[a-f0-9]{7,40}$/.test(sha)) {
    return respond({ ok: false, error: "invalid_build_sha" }, 400);
  }
  if (typeof body.engine_healthy !== "boolean") {
    return respond({ ok: false, error: "invalid_engine_healthy" }, 400);
  }
  const status = typeof body.http_status === "number" &&
    Number.isInteger(body.http_status) && body.http_status >= 0 && body.http_status <= 599
    ? body.http_status : null;
  const clean = (value: unknown) => {
    const v = typeof value === "string" ? value : "";
    return /^[a-zA-Z0-9_:-]{1,90}$/.test(v) ? v : null;
  };
  const blockers = Array.isArray(body.blockers)
    ? body.blockers.slice(0, 20).map(clean).filter((x): x is string => x !== null)
    : [];

  // Backward-compatible: old heartbeat clients omit these two fields.
  // Never accept strings/numbers as proof that a local service is reachable.
  const operatorStatus = (value: unknown): boolean | null =>
    typeof value === "boolean" ? value : null;
  for (const field of ["admin_http_responsive", "streamlit_http_responsive"]) {
    if (body[field] !== undefined && body[field] !== null &&
        typeof body[field] !== "boolean") {
      return respond({ ok: false, error: "invalid_operator_service_status" }, 400);
    }
  }

  const url = Deno.env.get("SUPABASE_URL") ?? "";
  const key = serviceKey();
  if (!url || !key) return respond({ ok: false, error: "server_configuration_missing" }, 500);
  const row = {
    source: "lightsail",
    observed_at: stamp,
    received_at: new Date().toISOString(),
    build_sha: sha || null,
    engine_healthy: body.engine_healthy,
    http_status: status,
    trading_status: clean(body.trading_status),
    blockers,
    failure_reason: clean(body.failure_reason),
    release_state: clean(body.release_state),
    admin_http_responsive: operatorStatus(body.admin_http_responsive),
    streamlit_http_responsive: operatorStatus(body.streamlit_http_responsive),
    release_updated_at: typeof body.release_updated_at === "string" &&
      Number.isFinite(Date.parse(body.release_updated_at)) &&
      Date.parse(body.release_updated_at) <= Date.now() + 300000
      ? new Date(body.release_updated_at).toISOString() : null,
  };
  const headers: Record<string,string> = {
    "content-type": "application/json",
    "apikey": key,
    "prefer": "resolution=merge-duplicates,return=minimal",
  };
  if (key.startsWith("eyJ")) headers.authorization = `Bearer ${key}`;
  try {
    const result = await fetch(
      `${url}/rest/v1/nifty_runtime_heartbeat?on_conflict=source`,
      { method: "POST", headers, body: JSON.stringify([row]) },
    );
    if (!result.ok) return respond({ ok: false, error: "heartbeat_write_failed" }, 500);
    return respond({ ok: true, observed_at: stamp });
  } catch {
    return respond({ ok: false, error: "heartbeat_store_unreachable" }, 503);
  }
});
