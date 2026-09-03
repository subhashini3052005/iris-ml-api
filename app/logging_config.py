import logging
from logging.handlers import RotatingFileHandler
from app.config import settings

def setup_logger():
    logger = logging.getLogger("ml_api")
    logger.setLevel(settings.LOG_LEVEL)

    if logger.handlers:
        return logger
    
    console_handler = logging.StreamHandler()

    file_handler = RotatingFileHandler(
        "app.log",
        maxBytes=1_000_000,
        backupCount=3
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    
    console_handler.setFormatter(formatter) 
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger