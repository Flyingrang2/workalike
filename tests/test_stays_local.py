"""What the product can reach. The README says the records and the model stay on the GPU you run it on, so this holds the
code to it: only two product files open a connection, every address written in the product is this machine, and the
model server answers only here and sends no usage reports."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NETWORK = {
    "socket",
    "requests",
    "httpx",
    "aiohttp",
    "http.client",
    "ftplib",
    "smtplib",
    "urllib.request",
    "websockets",
    "openai",
}
MAY_CONNECT = {"model_client.py", "serving.py"}  # the model server, at the address it is given; a check that it is up


def imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found |= {node.module} | {f"{node.module}.{a.name}" for a in node.names}
    return found


def test_only_the_model_client_and_the_server_check_open_connections():
    for path in (ROOT / "workalike").glob("*.py"):
        if path.name not in MAY_CONNECT:
            assert not imports(path) & NETWORK, path.name


def test_every_address_written_in_the_product_is_this_machine():
    for path in (ROOT / "workalike").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith(("http://", "https://"))
            ):
                # the bare scheme is the start of an f-string that names this machine's own host and port
                assert node.value == "http://" or node.value.startswith(("http://localhost", "http://127.0.0.1")), (
                    path.name,
                    node.value,
                )


def test_the_desk_points_at_the_model_on_this_machine():
    assert 'default="http://localhost:8000/v1"' in (ROOT / "workalike/__main__.py").read_text()


def test_the_model_server_stays_on_this_machine_and_reports_nothing():
    serve = (ROOT / "scripts/serve.sh").read_text()
    assert "--host 127.0.0.1" in serve
    assert "VLLM_NO_USAGE_STATS=1" in serve and "DO_NOT_TRACK=1" in serve
