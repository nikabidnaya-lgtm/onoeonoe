# Three-Day Trading Bot (Bybit Demo)

MVP-реализация ТЗ стратегии «Понедельник / Среда / Воскресенье» с использованием:
- **Pybit** (исполнение ордеров на Bybit Unified Trading API)
- **aiogram** (Telegram-команды и уведомления)
- APScheduler (недельное расписание)
- SQLAlchemy + SQLite (журнал сделок и сигналов)

## Важно по безопасности
1. Не храните ключи API в коде.
2. Используйте `.env` и выдавайте API-ключу права только на торговлю (без вывода).
3. Для старта используйте **testnet/demo**.

## Быстрый старт
```bash
cd trading_bot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# заполните BYBIT_API_KEY, BYBIT_API_SECRET, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
python main.py
```

## Структура
- `core/data_collector.py` — рыночные данные (цены, funding, OHLCV, spread)
- `core/analytics.py` — проверка условий для ПН/СР/ВС
- `core/executor.py` — открытие/закрытие позиций и установка SL/TP
- `modules/*.py` — сценарии торговых дней
- `telegram_bot.py` — команды `/status`, `/kill_all`, `/pause`, `/resume`, `/config`, `/logs`
- `core/scheduler.py` — cron-расписание в МСК
- `database.py` — таблицы `trades`, `signals`

## Ограничения текущего MVP
- Заглушки для `BTC.D` и ETF-потоков (`get_btc_dominance`, `get_etf_flows`) — нужно подключить внешний провайдер.
- Частичный TP2/трейлинг-стоп и расширенный backtest не реализованы в этой версии.
- Рекомендуется добавить интеграционные тесты и dry-run режим перед реальной торговлей.
