import logging

DEBUGFORMATTER = "%(filename)s:%(name)s:%(funcName)s:%(lineno)d: %(message)s"
"""Debug file formatter."""

INFOFORMATTER = "%(message)s"
"""Log file and stream output formatter."""

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# defines the stream handler
_ch = logging.StreamHandler()  # creates the handler
_ch.setLevel(logging.WARN)  # sets the handler info
_ch.setFormatter(
    logging.Formatter(DEBUGFORMATTER)
)  # sets the handler formatting

# adds the handler to the global variable: log
logger.addHandler(_ch)
