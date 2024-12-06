def test_api():
    from torchutils.eval import eval_wikitext2
    from torchutils.bench import bench_module, bench_more
    from torchutils import freeze_seed


def test_log_info():
    from torchutils import log_info

    log_info()
    import logging

    logger = logging.getLogger(__file__)
    test_msg = "This is a test message!!!!!!!!!!!"
    logger.info(test_msg)


def test_log_info2(caplog):
    import logging
    from torchutils import log_info

    with caplog.at_level(logging.INFO):
        log_info()
        assert "Set logging level to INFO successfully" in caplog.text
