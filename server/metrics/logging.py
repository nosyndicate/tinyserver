import json
import logging
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def log_event(
    event: str,
    *,
    log_level: int = logging.INFO,
    exc_info: bool = False,
    **fields: Any,
) -> None:
    """
    Log an event with structured fields as a JSON object.

    Args:
        event (str): The name of the event to log.
        **fields: Additional key-value pairs to include in the log entry.
    """
    log_entry = {"event": event, **fields}
    logger.log(
        log_level,
        json.dumps(log_entry, ensure_ascii=False),
        exc_info=exc_info,
    )
