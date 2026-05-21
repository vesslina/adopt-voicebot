# ISP Voicebot

Локальный голосовой AI-ассистент для интернет-провайдера.

Проект разрабатывается как тестовый прототип замены классического IVR-автоответчика формата "нажмите 1, чтобы узнать баланс". Вместо меню по кнопкам абонент разговаривает с голосовым ассистентом естественным языком. Звонок приходит в Asterisk, аудио передается в Python-сервер через AudioSocket, дальше обрабатывается локальными STT, LLM и TTS-компонентами.

Это не production-версия. В текущем состоянии проект демонстрирует архитектуру, голосовой пайплайн и базовые сценарии обслуживания абонента. Клиентские данные, баланс, платежи, заявки и перевод на оператора пока реализованы как mock-логика внутри Python-кода. Реальная база биллинга, CRM, SIP-транк и production-аутентификация еще не подключались.

## Текущий статус

Прототип уже работает end-to-end:

- Asterisk отправляет аудио звонка в Python-сервер через AudioSocket.
- Сервер определяет речь через Silero VAD.
- Речь распознается через faster-whisper.
- Частые бизнес-запросы могут обрабатываться напрямую, без ожидания LLM.
- Вопросы по теме интернет-провайдера отправляются в локальную LLM через Ollama.
- Ответ синтезируется через Silero TTS.
- Готовое аудио возвращается обратно в Asterisk.
- Абонент может перебивать ассистента во время ответа.
- Числа, суммы, даты и название компании нормализуются перед озвучкой.
- Для известного demo-абонента ассистент начинает звонок с короткого приветствия.

Облачные LLM, STT или TTS API не используются.

## Стек

- Телефония: Asterisk + AudioSocket
- Язык: Python 3.11
- VAD: Silero VAD
- STT: faster-whisper
- LLM runtime: Ollama
- Текущая локальная LLM: `qwen3:8b`
- TTS: Silero TTS `v5_5_ru`
- Упаковка: Docker Compose

## Архитектура

Планируемая production-схема:

```text
Linux-сервер
  ├─ Native Asterisk
  └─ Docker Compose
       ├─ voicebot Python service
       └─ Ollama local LLM runtime
```

Текущая тестовая схема:

```text
ПК 1: Ubuntu Server
  └─ Asterisk

ПК 2: Windows-ноутбук с Docker Desktop и NVIDIA GPU
  └─ Docker Compose
       ├─ voicebot Python service
       └─ Ollama with qwen3:8b
```

В обеих схемах Asterisk подключается к voicebot-сервису по TCP-порту `8090`.

## Реализованные функции

- AudioSocket TCP-сервер для приема аудиофреймов от Asterisk.
- Полный пайплайн звонка: VAD -> STT -> business logic или LLM -> TTS -> AudioSocket response.
- Keepalive-аудиофреймы для Asterisk во время долгой обработки запроса.
- Отдельный reader-поток, чтобы TCP-буфер AudioSocket не переполнялся.
- Barge-in: абонент может начать говорить, пока ассистент еще озвучивает ответ.
- Автоматическое приветствие при старте звонка.
- Контекст известного demo-абонента.
- Быстрые deterministic-ответы для типовых запросов:
  - баланс;
  - последние платежи;
  - статус заявки;
  - перевод к оператору;
  - базовая диагностика проблемы "не работает интернет".
- Строгий системный промпт, ограничивающий ассистента задачами интернет-провайдера.
- Нормализация текста перед русской озвучкой:
  - номера договоров;
  - суммы;
  - тарифы с числами;
  - даты;
  - латинское название компании.
- Рабочие `.env`, `.env.example` и `.env.docker`.
- Docker Compose стек `voicebot + ollama`.
- Скрипты запуска для PowerShell, Windows CMD и Linux shell.

## Структура репозитория

```text
.
├─ Dockerfile
├─ compose.yaml
├─ requirements.txt
├─ requirements.docker.txt
├─ run.bat
├─ scripts/
│  ├─ docker-start.*
│  ├─ docker-stop.*
│  ├─ docker-logs.*
│  └─ ollama-pull.*
└─ system/
   ├─ app/
   │  └─ bot_server.py
   ├─ diagnostics/
   │  ├─ echo_server.py
   │  ├─ stt_server.py
   │  ├─ test_tts_voices.py
   │  └─ vad_server.py
   ├─ models/
   └─ output/
```

Папки `system/models` и `system/output` не коммитятся. Локальные `.pt`-модели, WAV-файлы, Ollama-модели, `.env`, `venv` и внутренние заметки исключены из Git.

## Конфигурация

Локальная разработка на Windows использует `.env`.

Docker-запуск использует `.env.docker`.

Основные параметры:

```env
VOICEBOT_PORT=8090
VOICEBOT_OLLAMA_MODEL=qwen3:8b
VOICEBOT_TTS_MODEL_ID=v5_5_ru
VOICEBOT_TTS_SPEAKER=xenia
VOICEBOT_DEMO_CALLER_ID=1001
```

`.env.example` можно использовать как шаблон для локального `.env`.

Файлы `.env.example` и `.env.docker` не содержат секретов. В них лежат только параметры локального запуска: порты, адреса сервисов внутри Docker, названия моделей и настройки голосового пайплайна. Реальный `.env` с приватными токенами, паролями, ключами API, доступами к биллингу или SIP-учетками не должен попадать в Git.

## Docker-запуск

PowerShell:

```powershell
cd D:\voicebot
.\scripts\docker-start.ps1
.\scripts\ollama-pull.ps1 qwen3:8b
docker compose restart voicebot
.\scripts\docker-logs.ps1
```

Windows CMD:

```bat
cd /d D:\voicebot
scripts\docker-start.bat
scripts\ollama-pull.bat qwen3:8b
docker compose restart voicebot
scripts\docker-logs.bat
```

Linux:

```bash
cd /opt/voicebot
chmod +x scripts/*.sh
./scripts/docker-start.sh
./scripts/ollama-pull.sh qwen3:8b
docker compose restart voicebot
./scripts/docker-logs.sh
```

Успешный запуск выглядит так:

```text
Ollama OK, модели: ['qwen3:8b']
Прогрев модели qwen3:8b...
Модель прогрета ... можно звонить!
Bot server listening on 0.0.0.0:8090
```

## Подключение Asterisk

Минимальный пример dialplan:

```asterisk
exten => 9000,1,NoOp(ISP Voicebot)
 same => n,Answer()
 same => n,AudioSocket(40325ec5-b8be-4cf1-9f42-1ca7a36c0a18,VOICEBOT_HOST:8090)
 same => n,Hangup()
```

Если Asterisk и Docker работают на одном Linux-сервере, `VOICEBOT_HOST` обычно может быть `127.0.0.1`.

Если Asterisk работает на отдельной машине, `VOICEBOT_HOST` должен быть IP-адресом машины, где запущен Docker.

## Модели

В текущем прототипе используется `qwen3:8b`, потому что эту модель можно запустить на доступном тестовом железе. Финальная версия может использовать более крупную модель при наличии подходящего сервера с достаточным объемом RAM и VRAM.

Чтобы заменить модель Ollama:

```powershell
.\scripts\ollama-pull.ps1 qwen3:14b
```

Затем изменить:

```env
VOICEBOT_OLLAMA_MODEL=qwen3:14b
```

и перезапустить контейнер `voicebot`.

## Ограничения текущей версии

- Реальная база биллинга пока не подключена.
- Реальная CRM или ticket-система пока не подключена.
- Demo-абоненты захардкожены.
- Перевод к оператору пока представлен как business intent и требует production-настройки Asterisk.
- Текущая LLM выбрана под тестовое железо, а не под максимальное качество ответов.
- Текущий локальный TTS работает, но качество голоса можно улучшать.
- Для production нужны безопасность, мониторинг, логирование, политика хранения данных и интеграция с реальными системами компании.

## Ближайший план

- Проверить Docker-версию с реальным Asterisk-сервером.
- Подключить реальные данные биллинга.
- Подключить реальные заявки и CRM-действия.
- Добавить production-safe идентификацию абонента.
- Улучшить качество TTS.
- Настроить финальный dialplan для перевода к оператору.
- Подготовить deployment-инструкцию для одного Linux-сервера.

## English Summary

Local voice AI assistant prototype for an internet service provider. The project connects Asterisk AudioSocket with a Python voice pipeline, local faster-whisper STT, Ollama LLM, and Silero TTS. It is currently a test prototype with mock customer data and Docker packaging for the AI service stack.
