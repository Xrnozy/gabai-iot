import pytest


@pytest.fixture(scope="session", autouse=True)
def _rerun_warm():
    """The first rr.log in a process takes about 0.5 s (implicit recording start). In a headless test run it landed on the drawing thread in the middle of a burst of
    frames and made the pipeline tests drop panels; a real run has called rr.init by then. Pay it once, before any test."""
    import rerun as rr
    rr.log("test/warmup", rr.TextLog("warm"))
