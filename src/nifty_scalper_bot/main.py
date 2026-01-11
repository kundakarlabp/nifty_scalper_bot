import sys
import os

# 1. PRE-IMPORT LOGGING
print("🐍 PYTHON START: main.py loaded", flush=True)

try:
    print("⏳ Importing asyncio...", flush=True)
    import asyncio
    
    print("⏳ Importing logging...", flush=True)
    import logging
    
    print("⏳ Importing FastAPI...", flush=True)
    from fastapi import FastAPI
    from contextlib import asynccontextmanager
    
    print("⏳ Importing DotEnv...", flush=True)
    from dotenv import load_dotenv
    
    # 🛑 TRAP: Heavy imports often freeze deployments
    print("⏳ Importing Core App (Lazy Load check)...", flush=True)
    # We do NOT import NiftyScalperApp here. We import it inside lifespan.
    # If we import it here, and it has top-level blocking code, we die.
    print("✅ Imports Complete.", flush=True)

except Exception as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}", flush=True)
    sys.exit(1)

# --- SETUP ---
load_dotenv(override=True)
logging.basicConfig(level="INFO", stream=sys.stdout)
LOG = logging.getLogger("nifty_scalper_bot.main")

# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🟢 LIFESPAN: Startup Event Triggered", flush=True)
    
    # Background Task
    async def start_bot():
        print("🤖 BACKGROUND TASK: Waking up...", flush=True)
        try:
            print("⏳ BACKGROUND TASK: Sleeping 5s to allow port bind...", flush=True)
            await asyncio.sleep(5)
            
            print("📦 BACKGROUND TASK: Importing NiftyScalperApp...", flush=True)
            from nifty_scalper_bot.core.app import NiftyScalperApp
            
            print("🚀 BACKGROUND TASK: Initializing Bot...", flush=True)
            bot = NiftyScalperApp()
            app.state.bot = bot
            
            print("▶️ BACKGROUND TASK: Calling bot.start()...", flush=True)
            await bot.start()
            
        except Exception as exc:
            print(f"❌ BACKGROUND TASK DIED: {exc}", flush=True)
            LOG.error(f"Bot Crash: {exc}", exc_info=True)

    task = asyncio.create_task(start_bot())
    print("⚡ LIFESPAN: Yielding to Server...", flush=True)
    
    yield 
    
    print("🔴 LIFESPAN: Shutdown Triggered", flush=True)

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    print("📡 HTTP REQUEST: / root accessed", flush=True)
    return {"status": "ok"}
