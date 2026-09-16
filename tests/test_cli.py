import pytest

from migracoast_ml.__main__ import build_parser


def test_parser_exposes_every_pipeline_stage():
    parser = build_parser()
    for command in ("fetch", "build", "train", "evaluate", "baseline", "serve"):
        assert parser.parse_args([command]).command == command


def test_fetch_accepts_a_force_flag():
    assert build_parser().parse_args(["fetch", "--force"]).force is True


def test_fetch_defaults_to_not_forcing():
    assert build_parser().parse_args(["fetch"]).force is False


def test_serve_accepts_host_and_port():
    args = build_parser().parse_args(["serve", "--port", "8080"])
    assert args.port == 8080


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["nonsense"])
