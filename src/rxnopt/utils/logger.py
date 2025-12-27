from __future__ import annotations
from typing import Optional
import logging
from pathlib import Path
from rich.console import Console
from rich.theme import Theme
from rich.logging import RichHandler


class RXNConsole:
    """
    提供项目全局可复用的 rich Console 与 logging 配置。
    使用方式：
      console.print("hello")
    """

    def __init__(
        self,
        *,
        level: int = logging.INFO,
        logfile: Optional[Path | str] = None,
        force_terminal: bool = False,
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
        self.console = Console(
            theme=self.theme,
            force_terminal=force_terminal,
            quiet=False if level > 0 else True,
        )

        self._configure_logging(level=level, logfile=logfile)

    def _configure_logging(self, *, level: int, logfile: Optional[Path | str]) -> None:
        handler = RichHandler(
            console=self.console,
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True,
        )
        logging.basicConfig(level=level, handlers=[handler], format="%(message)s")
        logging.getLogger().setLevel(level)

        if logfile:
            log_path = Path(logfile)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
            logging.getLogger().addHandler(file_handler)

    def get_console(self) -> Console:
        return self.console

    def set_level(self, level: int) -> None:
        logging.getLogger().setLevel(level)


# module-level singleton console，整个项目直接 import 使用
_default = RXNConsole()
console: Console = _default.get_console()
