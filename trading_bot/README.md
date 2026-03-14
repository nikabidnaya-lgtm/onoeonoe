# Five-Day Trading Bot (Bybit Demo, Pybit + aiogram)

Реализация расширенного ТЗ стратегии **«5 дней»**:
- ПН: short (SOL, ETH, ARB, OP)
- ВТ: short (ETH, BNB, SOL, XRP)
- СР: trend (SOL, ETH, ARB, OP)
- ПТ: short (ETH, BNB, SOL, XRP)
- ВС: long (SOL, ETH, ARB, OP)

## Что исправлено относительно прошлого MVP
- Перевод конфигурации на `sessions.*` (каждый день отдельно: symbols, filters, sl/tp, execution).
- Добавлены отдельные стратегии `tuesday.py` и `friday.py`.
- Добавлены выходы по расписанию для каждой сессии (автозакрытие позиций).
- Добавлен контроль паузы бота (`/pause`, `/resume`) в стратегиях, а не только в Telegram-хендлерах.
- Добавлен параметр `/logs N` для выдачи последних N строк лога.
- Добавлен контроль проскальзывания перед market-входом.

## Быстрый старт
```bash
cd trading_bot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```

## Конфиг
Главный файл: `config.yaml`.
Ключи/секреты подставляются из `.env` через `${VAR_NAME}`.

## Важные ограничения
- `get_btc_dominance`, `get_etf_flows`, `get_economic_calendar` сейчас stub-методы: нужно подключить реальных провайдеров.
- Частичный TP1/TP2 и трейлинг реализованы как конфиг/база, но для полноценного много-ордерного исполнения потребуется отдельный order-manager.
- Перед live обязательны dry-run и проверка всех путей исполнения на testnet.
