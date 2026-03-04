import json
import logging
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def log_event(event: str, **fields: Any) -> None:
    """
    Log an event with structured fields as a JSON object.

    Args:
        event (str): The name of the event to log.
        **fields: Additional key-value pairs to include in the log entry.
    """
    log_entry = {"event": event, **fields}
    logger.info(json.dumps(log_entry, ensure_ascii=False))
