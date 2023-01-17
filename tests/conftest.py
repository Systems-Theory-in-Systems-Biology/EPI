def pytest_sessionstart(session):
    """Called once at the start of a pytest session

    Currently disables the matplotlib output
    """
    import matplotlib

    matplotlib.use(
        "Template"
    )  # Prevent plots from showing by changing graphics backend to null template
