# Шпаргалка по работе с базой знаний и расширению функционала

## 📁 Где хранится информация

### База знаний (Knowledge Base)
- **Путь**: `system/knowledge/`
- **Структура**:
  ```
  system/knowledge/
  ├── tariffs/
  │   ├── domashny-plus.md
  │   └── biznes-standart.md
  ├── services/
  │   ├── wireless.md
  │   └── cameras.md
  ├── faq/
  │   ├── no-internet.md
  │   └── payment-methods.md
  └── index.md
  ```

### Текущая информация в коде
- **Mock данные клиентов**: В `bot_server.py` в переменной `MOCK_CUSTOMERS` (строки ~146-169)
- **Маппинг caller_id → contract_id**: В `bot_server.py` в переменной `CALLER_TO_CONTRACT` (строки ~171-174)
- **Системный промпт**: В `bot_server.py` переменная `SYSTEM_PROMPT` (строки ~162-173)

## 🔧 Где заменить TODO на реальные реализации

### Tool calling функции (в `bot_server.py`)
Найдите функции с комментариями `TODO: заменить на реальный API/запрос к БД`:

1. **get_balance(contract_id)** - получить баланс по договору
   - Текущая реализация: возвращает данные из `MOCK_CUSTOMERS`
   - Реальная реализация: запрос к биллинговой системе/API

2. **get_request_status(contract_id)** - получить статус заявки
   - Текущая реализация: возвращает данные из `MOCK_CUSTOMERS`
   - Реальная реализация: запрос к CRM/тикет-системе

3. **get_subscriber_info(phone_number)** - получить информацию об абоненте по номеру телефона
   - Текущая реализация: обратный поиск в `MOCK_CUSTOMERS` и `CALLER_TO_CONTRACT`
   - Реальная реализация: запрос к Worknet API или аналогичной системе

4. **transfer_to_human(reason)** - перевод на оператора
   - Текущая реализация: логирование и возврат None
   - Реальная реализация: интеграция с Asterisk для реального перевода вызова

## 📝 Пример замены TODO на реальный API

### Было (заглушка):
```python
def get_balance(contract_id):
    """
    Get balance for a contract.
    TODO: заменить на реальный API/запрос к БД
    """
    customer = MOCK_CUSTOMERS.get(contract_id)
    if customer:
        return {"balance": customer["balance"], "currency": "RUB"}
    return {"balance": 0, "currency": "RUB"}
```

### Стало (пример с реальным API):
```python
def get_balance(contract_id):
    """
    Get balance for a contract by calling billing API.
    """
    try:
        # Пример запроса к реальному API
        response = requests.get(
            f"https://api.myprovider.ru/v1/billing/balance",
            params={"contract_id": contract_id},
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        return {
            "balance": data["balance"],
            "currency": data.get("currency", "RUB"),
            "last_updated": data.get("timestamp")
        }
    except requests.RequestException as e:
        log.error(f"Failed to get balance for contract {contract_id}: {e}")
        # Возвращаем заглушку или ошибку в зависимости от политики обработки ошибок
        return {"balance": 0, "currency": "RUB", "error": str(e)}
```

### Или с использованием внутреннего сервиса:
```python
def get_balance(contract_id):
    """
    Get balance for a contract by calling internal billing service.
    """
    try:
        # Предполагаем, что у нас есть внутренний сервис для работы с биллингом
        from billing_service import BillingService
        billing = BillingService()
        balance_data = billing.get_contract_balance(contract_id)
        return {
            "balance": balance_data.amount,
            "currency": balance_data.currency,
            "last_updated": balance_data.updated_at
        }
    except Exception as e:
        log.error(f"Failed to get balance for contract {contract_id}: {e}")
        return {"balance": 0, "currency": "RUB", "error": str(e)}
```

## 🚀 Как добавить новую информацию в базу знаний

1. Создайте новый .md файл в соответствующей подпапке:
   - Тарифы → `system/knowledge/tariffs/`
   - Услуги → `system/knowledge/services/`
   - Вопросы и ответы → `system/knowledge/faq/`

2. Заполните файл markdown-контентом с полезной информацией для абонентов

3. Обновите `system/knowledge/index.md` добавив запись о новом файле

4. Перезапустите голосового бота (или дождитесь автоматической перезагрузки при изменении файлов, если реализовано)

## ⚙️ Технические детали работы RAG

- При запуске сервера все .md файлы из `system/knowledge/` загружаются в память
- При получении запроса от абонента выполняется поиск по ключевым словам
- Находятся топ-2 наиболее релевантных документа
- Содержимое этих документов добавляется в системный промпт LLM перед каждым запросом
- Поиск реализован простым совпадением слов (без учета порядка, с приведением к нижнему регистру)
- Типичное время поиска: < 10 мс

## 📝 Пример работы в действии

**Абонент спрашивает**: "Как подключить беспроводной интернет на даче?"

**Процесс**:
1. Система извлекает ключевые слова: ["подключить", "беспроводной", "интернет", "даче"]
2. Ищет в базе знаний документы с совпадающими словами
3. Находит совпадения в `services/wireless.md` (беспроводной интернет)
4. Добавляет содержимое этого документа в контекст LLM
5. LLM отвечает на основе информации из документа + своих знаний

---

*Этот документ поможет вам ориентироваться в структуре проекта и быстро вносить изменения по мере развития голосового ассистента.*