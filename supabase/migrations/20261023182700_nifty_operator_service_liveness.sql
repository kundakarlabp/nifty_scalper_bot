-- Expand the existing private Lightsail heartbeat; no new table or RLS policy.
-- Missing readings from older GitHub releases intentionally remain NULL/unknown.
alter table public.nifty_runtime_heartbeat
    add column if not exists admin_http_responsive boolean,
    add column if not exists streamlit_http_responsive boolean;

comment on column public.nifty_runtime_heartbeat.admin_http_responsive
    is 'Independent Lightsail-loopback admin HTTP liveness; NULL means unknown';
comment on column public.nifty_runtime_heartbeat.streamlit_http_responsive
    is 'Independent Lightsail-loopback Streamlit HTTP liveness; NULL means unknown';
