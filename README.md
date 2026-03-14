# Bybit 5-day bot (single-file)

По вашему запросу бот собран в **одном Python-файле**: `trading_bot_single.py`.

## Запуск
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python trading_bot_single.py
```

## Конфиг
- `config.yaml` — параметры стратегий ПН/ВТ/СР/ПТ/ВС
- `.env.example` — переменные окружения для ключей

## Команды Telegram
- `/status`
- `/kill_all`
- `/pause`
- `/resume`
- `/config`
- `/logs N`

## Примечание
Docker-структура удалена: текущая версия специально упрощена под формат «один файл Python».
