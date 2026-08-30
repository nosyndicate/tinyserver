import pytest

from server.main import parse_args


def test_v4_defaults_use_single_worker_backlog() -> None:
    args = parse_args(["v4"])

    assert args.worker_queue_size == 128
    assert args.max_num_sequences == 8
    assert args.max_num_tokens == 4096
    assert not hasattr(args, "max_waiting")


def test_v4_rejects_removed_max_waiting_flag() -> None:
    with pytest.raises(SystemExit):
        parse_args(["v4", "--max-waiting", "64"])


@pytest.mark.parametrize(
    "argv",
    [
        ["v4", "--max-num-sequences", "0"],
        ["v4", "--max-num-sequences", "8", "--max-num-tokens", "7"],
    ],
)
def test_v4_rejects_invalid_scheduler_capacity_before_startup(
    argv: list[str],
) -> None:
    with pytest.raises(SystemExit):
        parse_args(argv)
