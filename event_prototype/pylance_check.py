#!/usr/bin/env python3
"""Type-check Python files with Pylance.

Pylance ships only as a stdio language server, so this is a small LSP client:
it starts the server, opens the requested files, collects the diagnostics it
publishes, and prints them.  The server path and settings mirror the ones in
~/.config/nvim/init.lua.
"""

from __future__ import annotations

import argparse
import json
import os
import selectors
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# pylance and node live at hashed store paths that move on every rebuild, so
# they are discovered rather than hardcoded.  editor.nix puts pylance on PATH
# only inside neovim's wrapper, hence the store glob.
def find_pylance() -> str:
    override = os.environ.get("PYLANCE_BIN")
    if override:
        return override
    found = shutil.which("pylance")
    if found:
        return found
    candidates = list(Path("/nix/store").glob("*-pylance-*/bin/pylance"))
    if not candidates:
        sys.exit(
            "pylance not found: set PYLANCE_BIN, or check that "
            "secrets/pylance.nix is present so editor.nix builds it"
        )

    def rank(path: Path):
        version = path.parent.parent.name.rsplit("-pylance-", 1)[-1]
        parts = tuple(int(p) if p.isdigit() else -1 for p in version.split("."))
        return (parts, path.stat().st_mtime)

    return str(max(candidates, key=rank))


def find_node() -> str:
    return shutil.which("node") or ""


# The license blob pylance demands from non-Microsoft clients, sent verbatim
# with its surrounding quotes.  Byte-identical to the JSON string literal that
# editor.nix reads out of secrets/pylance-license.json, kept inline so this
# script stands alone.
CLIENT_VERIFICATION = (
    '"You may install and use any number of copies of the software only with '
    "Microsoft Visual Studio, Visual Studio for Mac, Visual Studio Code, Azure "
    "DevOps, Team Foundation Server, and successor Microsoft products and "
    "services (collectively, the “Visual Studio Products and Services”) "
    "to develop and test your applications. The software is licensed, not sold. "
    "This agreement only gives you some rights to use the software. Microsoft "
    "reserves all other rights. You may not: work around any technical "
    "limitations in the software that only allow you to use it in certain ways; "
    "reverse engineer, decompile or disassemble the software, or otherwise "
    "attempt to derive the source code for the software, except and to the "
    "extent required by third party licensing terms governing use of certain "
    "open source components that may be included in the software; remove, "
    "minimize, block, or modify any notices of Microsoft or its suppliers in the "
    "software; use the software in any way that is against the law or to create "
    "or propagate malware; or share, publish, distribute, or lease the software "
    "(except for any distributable code, subject to the terms above), provide "
    "the software as a stand-alone offering for others to use, or transfer the "
    'software or this agreement to any third party."'
)

SEVERITY = {1: "error", 2: "warning", 3: "info", 4: "hint"}


def python_settings(
    type_checking_mode: str, diagnostic_mode: str, python_path: str
) -> dict:
    return {
        "python": {
            "pythonPath": python_path,
            "analysis": {
                "typeCheckingMode": type_checking_mode,
                "diagnosticMode": diagnostic_mode,
                "languageServerMode": "default",
                "nodeExecutable": find_node(),
                "autoImportCompletions": False,
                "useLibraryCodeForTypes": True,
            },
        }
    }


class Server:
    def __init__(self, cmd: list[str], cwd: str, debug: bool = False):
        self.debug = debug
        self.proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.buf = b""
        self.next_id = 0
        self.sel = selectors.DefaultSelector()
        self.sel.register(self.proc.stdout, selectors.EVENT_READ)
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stderr(self) -> None:
        for line in self.proc.stderr:
            if self.debug:
                sys.stderr.write("[pylance] " + line.decode(errors="replace"))

    def send(self, msg: dict) -> None:
        body = json.dumps(msg).encode()
        self.proc.stdin.write(b"Content-Length: %d\r\n\r\n" % len(body) + body)
        self.proc.stdin.flush()

    def notify(self, method: str, params: dict) -> None:
        self.send({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method: str, params: dict) -> int:
        self.next_id += 1
        self.send(
            {
                "jsonrpc": "2.0",
                "id": self.next_id,
                "method": method,
                "params": params,
            }
        )
        return self.next_id

    def reply(self, req_id, result) -> None:
        self.send({"jsonrpc": "2.0", "id": req_id, "result": result})

    def read(self, timeout: float) -> dict | None:
        """Next message, or None if nothing arrived within `timeout`."""
        deadline = time.monotonic() + timeout
        while True:
            msg = self._take()
            if msg is not None:
                return msg
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            if not self.sel.select(remaining):
                continue
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("pylance exited unexpectedly")
            self.buf += chunk

    def _take(self) -> dict | None:
        head, sep, rest = self.buf.partition(b"\r\n\r\n")
        if not sep:
            return None
        length = 0
        for line in head.split(b"\r\n"):
            name, _, value = line.partition(b":")
            if name.strip().lower() == b"content-length":
                length = int(value)
        if len(rest) < length:
            return None
        self.buf = rest[length:]
        return json.loads(rest[:length])

    def shutdown(self) -> None:
        try:
            self.request("shutdown", {})
            self.notify("exit", {})
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def check(
    files: list[Path],
    root: Path,
    type_checking_mode: str,
    diagnostic_mode: str,
    python_path: str,
    timeout: float,
    debug: bool,
) -> dict[str, list[dict]]:
    settings = python_settings(type_checking_mode, diagnostic_mode, python_path)
    server = Server([find_pylance()], cwd=str(root), debug=debug)
    wanted = {f.as_uri() for f in files}
    diagnostics: dict[str, list[dict]] = {}

    try:
        init_id = server.request(
            "initialize",
            {
                "processId": os.getpid(),
                "clientInfo": {"name": "pylance_check", "version": "1"},
                "rootUri": root.as_uri(),
                "workspaceFolders": [{"uri": root.as_uri(), "name": root.name}],
                "initializationOptions": {
                    "clientVerification": CLIENT_VERIFICATION,
                },
                "capabilities": {
                    "workspace": {
                        "configuration": True,
                        "workspaceFolders": True,
                        "didChangeConfiguration": {"dynamicRegistration": True},
                    },
                    "textDocument": {
                        "synchronization": {"dynamicRegistration": True},
                        "publishDiagnostics": {
                            "relatedInformation": True,
                            "versionSupport": True,
                        },
                    },
                    "window": {"workDoneProgress": True},
                },
            },
        )

        changed_at = time.monotonic()

        def lookup(section: str):
            value = settings
            for part in section.split("."):
                if not part:
                    continue
                value = value.get(part, {}) if isinstance(value, dict) else {}
            return value

        def handle(msg: dict) -> None:
            """Record diagnostics and answer whatever the server asks of us."""
            nonlocal changed_at
            method = msg.get("method")
            if method == "textDocument/publishDiagnostics":
                uri = msg["params"]["uri"]
                if uri in wanted:
                    new = msg["params"]["diagnostics"]
                    if diagnostics.get(uri) != new:
                        changed_at = time.monotonic()
                    diagnostics[uri] = new
            elif "id" in msg and method is not None:  # server -> client request
                if method == "workspace/configuration":
                    server.reply(
                        msg["id"],
                        [lookup(i.get("section") or "") for i in msg["params"]["items"]],
                    )
                else:
                    server.reply(msg["id"], None)

        def pump(deadline: float, until_id: int | None = None):
            """Handle traffic until `until_id` is answered or time runs out."""
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                msg = server.read(remaining)
                if msg is None:
                    return None
                if "id" in msg and "method" not in msg:
                    if until_id is not None and msg["id"] == until_id:
                        return msg
                    continue
                handle(msg)

        deadline = time.monotonic() + timeout
        if pump(deadline, until_id=init_id) is None:
            raise RuntimeError("pylance did not answer initialize in time")

        server.notify("initialized", {})
        server.notify("workspace/didChangeConfiguration", {"settings": settings})

        for path in files:
            server.notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": path.as_uri(),
                        "languageId": "python",
                        "version": 1,
                        "text": path.read_text(encoding="utf-8"),
                    }
                },
            )

        # Diagnostics arrive as pushes; stop once every file has reported and
        # the server has gone quiet, since it may revise its first answer.
        quiet = 2.0
        changed_at = time.monotonic()
        while time.monotonic() < deadline:
            if len(diagnostics) == len(wanted) and time.monotonic() - changed_at > quiet:
                break
            msg = server.read(0.25)
            if msg is not None:
                handle(msg)

        missing = wanted - set(diagnostics)
        if missing:
            for uri in missing:
                print(f"warning: no diagnostics reported for {uri}", file=sys.stderr)
        return diagnostics
    finally:
        server.shutdown()


def report(diagnostics: dict[str, list[dict]], root: Path) -> int:
    counts: dict[str, int] = {}
    for uri in sorted(diagnostics):
        path = Path(uri.removeprefix("file://"))
        try:
            shown = path.relative_to(root)
        except ValueError:
            shown = path
        for diag in sorted(
            diagnostics[uri],
            key=lambda d: (d["range"]["start"]["line"], d["range"]["start"]["character"]),
        ):
            sev = SEVERITY.get(diag.get("severity", 1), "error")
            counts[sev] = counts.get(sev, 0) + 1
            line = diag["range"]["start"]["line"] + 1
            col = diag["range"]["start"]["character"] + 1
            rule = diag.get("code")
            suffix = f" ({rule})" if rule else ""
            message = diag["message"].replace("\n", "\n    ")
            print(f"{shown}:{line}:{col}: {sev}: {message}{suffix}")

    summary = ", ".join(f"{n} {name}{'s' if n != 1 else ''}" for name, n in counts.items())
    print(summary or "no problems found")
    return 1 if counts.get("error") else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--root",
        type=Path,
        help="workspace root (default: nearest dir with pyproject.toml, else cwd)",
    )
    parser.add_argument(
        "--mode",
        default="strict",
        choices=["off", "basic", "standard", "strict"],
        help="python.analysis.typeCheckingMode (default: strict)",
    )
    parser.add_argument(
        "--workspace",
        action="store_true",
        help="analyze the whole workspace instead of just the given files",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter whose environment supplies imports (default: this one)",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--debug", action="store_true", help="show server stderr")
    args = parser.parse_args()

    files = [f.resolve() for f in args.files]
    for f in files:
        if not f.is_file():
            parser.error(f"not a file: {f}")

    if args.root:
        root = args.root.resolve()
    else:
        root = Path.cwd().resolve()
        for candidate in [files[0].parent, *files[0].parents]:
            if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
                root = candidate
                break

    diagnostics = check(
        files,
        root,
        args.mode,
        "workspace" if args.workspace else "openFilesOnly",
        args.python,
        args.timeout,
        args.debug,
    )
    return report(diagnostics, root)


if __name__ == "__main__":
    sys.exit(main())
