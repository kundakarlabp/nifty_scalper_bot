const ALLOWED_IPS = new Set(['65.0.255.84'])

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  })
}

function clientIp(req: Request): string {
  const cf = req.headers.get('cf-connecting-ip')?.trim()
  if (cf) return cf
  const forwarded = req.headers.get('x-forwarded-for')
  if (forwarded) return forwarded.split(',')[0].trim()
  return req.headers.get('x-real-ip')?.trim() ?? ''
}

function adminKey(): string {
  try {
    const keys = JSON.parse(Deno.env.get('SUPABASE_SECRET_KEYS') ?? '{}')
    if (typeof keys.default === 'string' && keys.default) return keys.default
  } catch {}
  return Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
}


// Allow only bounded, source-derived diagnostic fields into Supabase JSONB.
function cleanStartupAudit(raw: unknown): Record<string, unknown> | null {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null
  const input = raw as Record<string, unknown>
  if (input.schema_version !== 1 ||
      input.protective_stop_verification !== 'not_established_from_log_markers') {
    return null
  }
  const counters = [
    'observed_startup_count', 'ws_started_count', 'reconcile_complete_count',
    'live_ready_count', 'broker_order_reconcile_ok_count',
    'startups_after_0930_count', 'startups_without_live_ready_count',
  ]
  const safe: Record<string, unknown> = {
    schema_version: 1,
    protective_stop_verification: 'not_established_from_log_markers',
  }
  for (const name of counters) {
    const value = input[name]
    if (typeof value !== 'number' || !Number.isSafeInteger(value) ||
        value < 0 || value > 1000000) return null
    safe[name] = value
  }
  // Phase 9 fields are optional so Phase 7 senders remain supported.
  for (const name of [
    'observed_repeat_startup_count', 'market_window_startup_count',
    'stop_armed_event_count', 'stop_unverified_event_count',
    'stop_not_armed_event_count', 'stop_ratchet_confirmed_event_count',
  ]) {
    if (input[name] === undefined) continue
    const count = input[name]
    if (typeof count !== 'number' || !Number.isSafeInteger(count) ||
        count < 0 || count > 1000000) return null
    safe[name] = count
  }
  if (input.restart_cause !== undefined) {
    if (input.restart_cause !== 'unknown_without_process_and_deployment_events') {
      return null
    }
    safe.restart_cause = input.restart_cause
  }
  for (const name of [
    'first_startup_at_ist', 'last_startup_at_ist', 'last_live_ready_at_ist',
  ]) {
    const value = input[name]
    if (value !== null && (typeof value !== 'string' ||
        !/^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d$/.test(value))) return null
    safe[name] = value
  }
  // Optional fields must not reject previously deployed archive senders.
  if (input.first_live_ready_at_ist !== undefined) {
    const first = input.first_live_ready_at_ist
    if (first !== null && (typeof first !== 'string' ||
        !/^(?:[01]\\d|2[0-3]):[0-5]\\d:[0-5]\\d$/.test(first))) return null
    safe.first_live_ready_at_ist = first
  }
  if (input.opening_window_ready_event_observed !== undefined) {
    if (typeof input.opening_window_ready_event_observed !== 'boolean') return null
    safe.opening_window_ready_event_observed = input.opening_window_ready_event_observed
  }
  const lag = input.max_startup_to_ready_seconds
  if (lag !== null && (typeof lag !== 'number' ||
      !Number.isSafeInteger(lag) || lag < 0 || lag > 86400)) return null
  safe.max_startup_to_ready_seconds = lag
  return safe
}

Deno.serve(async (req: Request) => {
  if (req.method !== 'POST') return json({ ok: false, error: 'method_not_allowed' }, 405)
  if (!ALLOWED_IPS.has(clientIp(req))) return json({ ok: false, error: 'forbidden_source' }, 403)
  if (req.headers.get('x-nifty-source') !== 'lightsail') {
    return json({ ok: false, error: 'invalid_source_header' }, 403)
  }

  let body: Record<string, unknown>
  try { body = await req.json() } catch { return json({ ok: false, error: 'invalid_json' }, 400) }

  const tradingDate = String(body.trading_date ?? '')
  const logText = String(body.log_text ?? '')
  if (!/^\d{4}-\d{2}-\d{2}$/.test(tradingDate) || !logText) {
    return json({ ok: false, error: 'invalid_archive_payload' }, 400)
  }

  const url = Deno.env.get('SUPABASE_URL') ?? ''
  const key = adminKey()
  if (!url || !key) return json({ ok: false, error: 'server_configuration_missing' }, 500)

  // Older senders omit the new summary. Do not clear previous evidence.
  const candidateAudit = body.startup_audit
  const startupAudit = candidateAudit === undefined
    ? undefined : cleanStartupAudit(candidateAudit)
  if (candidateAudit !== undefined && startupAudit === null) {
    return json({ ok: false, error: 'invalid_startup_audit' }, 400)
  }

  const row: Record<string, unknown> = {
    trading_date: tradingDate,
    captured_at: body.captured_at,
    source: body.source ?? 'lightsail-journald',
    build_sha: body.build_sha ?? null,
    line_count: Number(body.line_count ?? 0),
    byte_count: Number(body.byte_count ?? 0),
    log_text: logText,
    first_log_at: body.first_log_at ?? null,
    last_log_at: body.last_log_at ?? null,
    updated_at: new Date().toISOString(),
    finalized: Boolean(body.finalized),
    archive_version: 1,
  }
  if (startupAudit) row.summary = { startup_audit: startupAudit }
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    apikey: key,
    prefer: 'resolution=merge-duplicates,return=minimal',
  }
  if (key.startsWith('eyJ')) headers.authorization = `Bearer ${key}`

  const response = await fetch(
    `${url}/rest/v1/nifty_daily_market_logs?on_conflict=trading_date`,
    { method: 'POST', headers, body: JSON.stringify([row]) },
  )
  if (!response.ok) {
    return json({ ok: false, error: 'upsert_failed', detail: await response.text() }, 500)
  }
  return json({ ok: true, trading_date: tradingDate, finalized: row.finalized })
})
