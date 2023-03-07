DISABLE_PLOTS = True
LOGGING_LEVEL = "INFO"


def pytest_sessionstart(session):
    """Called once at the start of a pytest session

    Currently disables the matplotlib output

    Args:
      session:

    Returns:

    """
    import matplotlib

    if DISABLE_PLOTS:
        matplotlib.use(
            "Template"
        )  # Prevent plots from showing by changing graphics backend to null template

    import logging

    logging.getLogger("epi").setLevel(LOGGING_LEVEL)
