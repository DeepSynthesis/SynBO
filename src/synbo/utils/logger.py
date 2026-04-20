from __future__ import annotations
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.theme import Theme
from rich.logging import RichHandler


class RXNConsole:
    """
    Provides globally reusable rich Console and logging configuration for the project.
    Usage:
      console.print("hello")
    """

    def __init__(
        self,
        *,
        quiet: bool = False,
        logfile: Optional[Path | str] = None,
        force_terminal: bool = True,
    ):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self.theme = Theme(
            {
                "info": "cyan",
                "warning": "yellow",
                "error": "bold red",
                "success": "green",
                "path": "magenta",
            }
        )
        self.console = Console(theme=self.theme, force_terminal=force_terminal, quiet=quiet, color_system="auto")

    def get_console(self) -> Console:
        return self.console

    def _set_quiet(self, quiet: bool) -> None:
        self.console.quiet = quiet


_logger_default = RXNConsole()
console: Console = _logger_default.get_console()
