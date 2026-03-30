import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Path = Path("logs"),
    log_name: str = "motion_s_text_to_sign.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> None:
    """
    Set up logging for the Motion-S-Text-to-Sign Project

    - Logs to console (INFO+ by default)
    - Logs to a rotating file (DEBUG+ by default, up to ~5 MB * 3)
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_name

    # Avoid duplicate handlers if called multiple times
    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(logging.DEBUG)  # root collects everything; handlers filter

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # Rotating file handler
    fh = RotatingFileHandler(
        log_path,
        maxBytes=5_000_000,  # ~5 MB
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(file_level)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    root.addHandler(ch)
    root.addHandler(fh)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Convenience wrapper around logging.getLogger.
    Call setup_logging() once at program start, then use get_logger(__name__).
    """
    return logging.getLogger(name)
