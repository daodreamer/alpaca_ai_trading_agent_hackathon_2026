"""Environment loading for credentials.

The researcher lives inside a repository whose secrets sit in a root-level
``.env.local``, one directory up from this project. Rather than requiring every
invocation to export variables by hand, this module walks upward from the working
directory and loads the first ``.env.local`` and ``.env`` it finds.

Precedence, strictest first:

1.  A variable already present in the real environment. Always wins -- an
    explicit ``export`` or a CI secret must never be silently overridden by a
    file someone forgot about.
2.  ``.env.local``  -- personal, gitignored, holds real keys.
3.  ``.env``        -- shared defaults, may be committed.

Values are never logged. :func:`describe` exists precisely so that "is the key
loaded" can be answered without anyone printing the key to a terminal, a CI log,
or a screen recording.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ENV_FILES", "credential", "describe", "load_env_files"]

ENV_FILES = (".env.local", ".env")

# Where each provider's key is expected to live.
_KEY_VARIABLES: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _parse(text: str) -> dict[str, str]:
    """Parse a dotenv file. Deliberately small and boring.

    Supports ``KEY=value``, ``export KEY=value``, quoted values and ``#``
    comments. It does not support interpolation or multi-line values: a config
    format with surprises in it is a config format that leaks credentials into
    the wrong variable.
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def _search_upward(start: Path, name: str, limit: int = 6) -> Path | None:
    current = start.resolve()
    for _ in range(limit):
        candidate = current / name
        if candidate.is_file():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def load_env_files(start: Path | str | None = None, *, override: bool = False) -> list[Path]:
    """Load dotenv files into ``os.environ``. Returns the files actually read.

    ``override=False`` (the default) leaves existing environment variables alone.
    Pass ``True`` only when you genuinely want the file to win, which in practice
    means a test fixture and nothing else.
    """
    root = Path(start) if start else Path.cwd()
    loaded: list[Path] = []
    merged: dict[str, str] = {}

    # Merge the files first, lowest precedence to highest, and only then touch
    # os.environ. Applying them one file at a time would make the *first* file
    # win, because after it the key is already in the environment and the
    # "don't override" rule protects it from the file that should have won.
    for name in reversed(ENV_FILES):
        path = _search_upward(root, name)
        if path is None:
            continue
        merged.update(_parse(path.read_text(encoding="utf-8")))
        loaded.append(path)

    for key, value in merged.items():
        if override or key not in os.environ:
            os.environ[key] = value
    loaded.reverse()  # report highest precedence first
    return loaded


@dataclass(frozen=True, slots=True)
class CredentialStatus:
    """What is known about a credential, with nothing sensitive in it."""

    provider: str
    variable: str
    present: bool
    length: int

    def __str__(self) -> str:
        if not self.present:
            return f"{self.provider}: {self.variable} not set"
        return f"{self.provider}: {self.variable} set ({self.length} chars)"


def credential(provider: str, *, load: bool = True) -> str:
    """The API key for ``provider``, or a message explaining where to put one."""
    variable = _KEY_VARIABLES.get(provider)
    if variable is None:
        raise KeyError(f"unknown provider {provider!r}; known: {sorted(_KEY_VARIABLES)}")
    if load:
        load_env_files()
    value = os.environ.get(variable, "").strip()
    if not value:
        raise RuntimeError(
            f"{variable} is not set. Put it in .env.local at the repository root, "
            f"or export it: {variable}=..."
        )
    return value


def describe(*, load: bool = True) -> list[CredentialStatus]:
    """Which provider credentials are available. Reports lengths, never values."""
    if load:
        load_env_files()
    return [
        CredentialStatus(
            provider=provider,
            variable=variable,
            present=bool(os.environ.get(variable, "").strip()),
            length=len(os.environ.get(variable, "").strip()),
        )
        for provider, variable in sorted(_KEY_VARIABLES.items())
    ]
