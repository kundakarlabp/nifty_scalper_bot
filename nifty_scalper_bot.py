
import os
import time
import logging
import threading
import datetime
from telegram import Bot
from telegram.ext import Updater, CommandHandler
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

running = True

def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="🤖 NiftyNinja Bot started!")

def status(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="✅ Bot is running." if running else "⛔ Bot is stopped.")

def shutdown(update, context):
    global running
    running = False
    context.bot.send_message(chat_id=update.effective_chat.id, text="🛑 Bot shutdown triggered.")

def telegram_bot():
    updater = Updater(token=BOT_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(CommandHandler('status', status))
    dp.add_handler(CommandHandler('shutdown', shutdown))
    updater.start_polling()
    return updater

def trading_loop():
    global running
    while running:
        now = datetime.datetime.now().time()
        if now >= datetime.time(9, 15) and now <= datetime.time(15, 30):
            print("Running trading logic...")
        else:
            print("Market closed.")
        time.sleep(60)

def main():
    updater = telegram_bot()
    trade_thread = threading.Thread(target=trading_loop)
    trade_thread.start()
    updater.idle()

if __name__ == '__main__':
    main()
