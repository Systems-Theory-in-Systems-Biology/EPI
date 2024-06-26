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

    logging.getLogger("eulerpi").setLevel(LOGGING_LEVEL)

    # Seed the numpy random number generator to ensure reproducibility
    import numpy as np

    np.random.seed(0)
