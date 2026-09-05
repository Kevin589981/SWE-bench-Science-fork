#!/usr/bin/env python3
"""Run Pier with a port-aware filtered egress proxy.

Pier's normal filtered egress proxy only permits destination ports 80 and 443.
AvaCore allocates one isolated model proxy port per rollout, so this wrapper
widens Squid's *destination-port* ACL while retaining Pier's authenticated
allowlisted-domain policy.  It patches the generated compose asset in memory;
the installed Pier package is left untouched.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path


_PORT_RANGE = "1024-65535"


def _allow_dynamic_destination_ports(script_path: Path):
    script_path = Path(script_path)
    text = script_path.read_text(encoding="utf-8")
    text, ssl_count = re.subn(
        r"^acl SSL_ports port 443$",
        f"acl SSL_ports port 443 {_PORT_RANGE}",
        text,
        flags=re.MULTILINE,
    )
    text, safe_count = re.subn(
        r"^acl Safe_ports port 80 443$",
        f"acl Safe_ports port 80 443 {_PORT_RANGE}",
        text,
        flags=re.MULTILINE,
    )
    if ssl_count != 1 or safe_count != 1:
        raise RuntimeError(
            "unexpected Pier Squid template; refusing to run with an unverified "
            "filtered-egress policy"
        )
    script_path.write_text(text, encoding="utf-8")
    return script_path


def _patch_pier() -> None:
    setup = importlib.import_module("pier.environments.agent_setup")
    docker = importlib.import_module("pier.environments.docker.docker")
    original = setup.write_docker_proxy_compose

    def wrapped(*args, **kwargs):
        compose_path = original(*args, **kwargs)
        proxy_dir = kwargs.get("proxy_dir")
        if proxy_dir is None:
            raise RuntimeError("Pier did not provide the egress proxy directory")
        _allow_dynamic_destination_ports(Path(proxy_dir) / "start-squid.sh")
        return compose_path

    # docker.py imports this function directly, so patch both module bindings.
    setup.write_docker_proxy_compose = wrapped
    docker.write_docker_proxy_compose = wrapped


def main() -> None:
    _patch_pier()
    cli = importlib.import_module("pier.cli.main")
    cli.app()


if __name__ == "__main__":
    main()
