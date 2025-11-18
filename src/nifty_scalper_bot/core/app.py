    @app.on_event("startup")
    async def _warm_instrument_resolver_on_startup() -> None:
        try:
            import inspect
            ctx = get_latest_bot_context()
            if ctx is None:
                LOGGER.debug(
                    "Resolver warm skipped: no bot context available",
                    extra={"event": "resolver_warm.no_context"},
                )
                return

            resolver = getattr(ctx, "instrument_resolver", None) or getattr(
                ctx, "resolver", None
            )
            if resolver is None:
                LOGGER.debug(
                    "Resolver warm skipped: no resolver available on context",
                    extra={"event": "resolver_warm.no_resolver"},
                )
                return

            # Prefer explicit warm_from_broker_dump when available.
            warm_fn = getattr(resolver, "warm_from_broker_dump", None)
            loop = asyncio.get_running_loop()

            def _fn_needs_rows(fn: Callable[..., Any]) -> bool:
                try:
                    sig = inspect.signature(fn)
                    return len(sig.parameters) > 0
                except Exception:
                    return True

            if callable(warm_fn):
                # try to load rows from DB/CSV hints on ctx (non-fatal)
                rows: list[Mapping[str, Any]] = []
                conn: sqlite3.Connection | None = None
                try:
                    db_path = getattr(ctx, "instrument_db_path", None) or getattr(
                        ctx, "instrument_cache_db", None
                    ) or getattr(ctx, "instrument_csv_path", None)
                    if db_path:
                        conn = ensure_sqlite(str(db_path))
                        rows = load_rows_for_resolver(conn) or []
                except Exception as exc:
                    LOGGER.debug(
                        "Failed to load resolver rows from DB cache: %s",
                        exc,
                        extra={"event": "resolver_warm.load_rows_failed"},
                    )
                finally:
                    if conn is not None:
                        with suppress(Exception):
                            conn.close()

                needs_rows = _fn_needs_rows(warm_fn)
                try:
                    if inspect.iscoroutinefunction(warm_fn):
                        if needs_rows:
                            await warm_fn(rows)
                        else:
                            await warm_fn()
                    else:
                        if needs_rows:
                            await loop.run_in_executor(None, lambda: warm_fn(rows))
                        else:
                            await loop.run_in_executor(None, warm_fn)
                    LOGGER.info(
                        "InstrumentResolver warmed via warm_from_broker_dump on HTTP startup",
                        extra={"event": "resolver_warm.startup_success"},
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "InstrumentResolver warm_from_broker_dump failed on startup: %s",
                        exc,
                        extra={"event": "resolver_warm.startup_failed"},
                        exc_info=exc,
                    )
            else:
                # Fallback to generic warm() call with same safety
                generic_warm = getattr(resolver, "warm", None)
                try:
                    if callable(generic_warm):
                        if inspect.iscoroutinefunction(generic_warm):
                            await generic_warm()
                        else:
                            await loop.run_in_executor(None, generic_warm)
                        LOGGER.info(
                            "InstrumentResolver warmed via warm() fallback on HTTP startup",
                            extra={"event": "resolver_warm.startup_success_fallback"},
                        )
                except Exception as exc:
                    LOGGER.warning(
                        "InstrumentResolver warm() failed on startup: %s",
                        exc,
                        extra={"event": "resolver_warm.startup_failed_fallback"},
                        exc_info=exc,
                    )
        except Exception:
            LOGGER.exception(
                "Unexpected error while attempting to warm InstrumentResolver on startup",
                extra={"event": "resolver_warm.unexpected_error"},
            )
