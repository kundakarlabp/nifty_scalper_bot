from pathlib import Path

path = Path("src/nifty_scalper_bot/core/app.py")
text = path.read_text(encoding="utf-8")
old = '''    if current_options and (not selected_ce or not selected_pe):
        LOGGER.warning(
            "ACTIVE_DYNAMIC_BASKET_DEFERRED reason=selected_option_resolution_failed option_count=%d selected_ce=%s selected_pe=%s",
            len(current_options),
            selected_ce,
            selected_pe,
            extra={
                "event": "ACTIVE_DYNAMIC_BASKET_DEFERRED",
                "reason": "selected_option_resolution_failed",
                "option_count": len(current_options),
                "selected_ce": selected_ce,
                "selected_pe": selected_pe,
            },
        )
        return cast(str | None, old_ce), cast(str | None, old_pe)
    ctx.selected_ce = str(selected_ce) if selected_ce else None
    ctx.selected_pe = str(selected_pe) if selected_pe else None
'''
new = '''    if current_options and (not selected_ce or not selected_pe):
        LOGGER.warning(
            "ACTIVE_DYNAMIC_BASKET_DEFERRED reason=selected_option_resolution_failed option_count=%d selected_ce=%s selected_pe=%s",
            len(current_options),
            selected_ce,
            selected_pe,
            extra={
                "event": "ACTIVE_DYNAMIC_BASKET_DEFERRED",
                "reason": "selected_option_resolution_failed",
                "option_count": len(current_options),
                "selected_ce": selected_ce,
                "selected_pe": selected_pe,
            },
        )
        return cast(str | None, old_ce), cast(str | None, old_pe)

    # Selection authority changes only after the selected pair has proven
    # live Runner/DataHub delivery. Candidate symbols may be wired before
    # commit, but the previous selected pair remains authoritative until
    # both new edges are ready.
    previous_token_map = dict(getattr(ctx, "active_symbol_tokens", {}) or {})
    candidate_token_map = dict(basket.get("token_by_symbol") or {})
    if not candidate_token_map:
        candidate_token_map = resolve_active_basket_tokens(
            ctx,
            list(
                dict.fromkeys(
                    [
                        *current_symbols,
                        *current_options,
                        *[s for s in (selected_ce, selected_pe) if s],
                    ]
                )
            ),
            selected_ce,
            selected_pe,
        )
    runtime_delivery_enforceable = bool(
        getattr(ctx, "strategy_runner", None) is not None
        and getattr(ctx, "data_hub", None) is not None
        and getattr(ctx, "market_data_manager", None) is not None
    )
    if runtime_delivery_enforceable and selected_ce and selected_pe:
        if not (
            candidate_token_map.get(selected_ce)
            and candidate_token_map.get(selected_pe)
        ):
            LOGGER.warning(
                "ACTIVE_DYNAMIC_BASKET_DEFERRED reason=selected_option_token_missing selected_ce=%s selected_pe=%s",
                selected_ce,
                selected_pe,
                extra={
                    "event": "ACTIVE_DYNAMIC_BASKET_DEFERRED",
                    "reason": "selected_option_token_missing",
                    "selected_ce": selected_ce,
                    "selected_pe": selected_pe,
                },
            )
            return cast(str | None, old_ce), cast(str | None, old_pe)
        ctx.active_symbol_tokens = candidate_token_map
        delivery = _ensure_selected_option_runtime_delivery(
            ctx,
            selected_ce=selected_ce,
            selected_pe=selected_pe,
            reason="dynamic_basket_precommit",
        )
        if not (delivery.get(selected_ce) and delivery.get(selected_pe)):
            ctx.active_symbol_tokens = previous_token_map
            LOGGER.warning(
                "ACTIVE_DYNAMIC_BASKET_DEFERRED reason=selected_option_runner_delivery_missing selected_ce=%s selected_pe=%s delivery=%s",
                selected_ce,
                selected_pe,
                delivery,
                extra={
                    "event": "ACTIVE_DYNAMIC_BASKET_DEFERRED",
                    "reason": "selected_option_runner_delivery_missing",
                    "selected_ce": selected_ce,
                    "selected_pe": selected_pe,
                    "delivery": dict(delivery),
                },
            )
            return cast(str | None, old_ce), cast(str | None, old_pe)

    ctx.selected_ce = str(selected_ce) if selected_ce else None
    ctx.selected_pe = str(selected_pe) if selected_pe else None
'''
if text.count(old) != 1:
    raise SystemExit(f"expected exactly one basket commit target, found {text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
