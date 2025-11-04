from nifty_scalper_bot.utils.sampled_logging import RegimeSampler

# Global singleton sampler for app-wide system/diagnostic events
system_sampler: RegimeSampler = RegimeSampler(period_s=15.0, event_dedupe_s=5.0)

__all__ = ["system_sampler"]
