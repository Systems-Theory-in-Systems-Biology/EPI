def pytest_sessionstart(session):
    import matplotlib

    matplotlib.use(
        "Template"
    )  # Prevent plots from showing by changing graphics backend to null template
