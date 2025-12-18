import logging
import logging.config
import os
from logging.handlers import RotatingFileHandler

# Путь к директории с логами (создаём, если нет)
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "heart_risk_service.log")

# Конфигурация логирования в формате dictConfig
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "WARNING",
            "formatter": "detailed",
            "filename": LOG_FILE,
            "maxBytes": 5 * 1024 * 1024,  # 5 МБ
            "backupCount": 5,
            "encoding": "utf-8"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}

def setup_logging():
    """
    Функция настройки логирования.
    Вызывается один раз при старте приложения (в app.py).
    """
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    logger.info("Логирование успешно настроено. Лог-файл: %s", LOG_FILE)

# Дополнительно: удобная функция для получения логгера в других модулях
def get_logger(name: str):
    """Возвращает настроенный logger по имени модуля."""
    return logging.getLogger(name)