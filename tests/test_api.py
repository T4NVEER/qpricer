import qpricer


def test_version() -> None:
    assert qpricer.__version__


def test_public_api_exports() -> None:
    for name in qpricer.__all__:
        assert getattr(qpricer, name, None) is not None
