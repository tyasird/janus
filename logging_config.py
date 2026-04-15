
import sys
import emoji
from loguru import logger as global_logger
from typing import TYPE_CHECKING

pkg_logger = global_logger.bind(name="janus")


class CustomLogger:
    def __init__(self, logger):
        self._log = logger
        self._define_custom_levels()
        self._configure_output()

    def _safe_set_level(self, name: str, **kwargs):
        try:
            self._log.level(name, **kwargs)
        except ValueError:
            pass  # Level already exists, ignore

    def _define_custom_levels(self):
        self._safe_set_level("DONE", no=25, color="<green>", icon=emoji.emojize(":flexed_biceps:"))
        self._safe_set_level("PROGRESS", no=22, color="<red>", icon=emoji.emojize(":horse:"))
        self._safe_set_level("STARTED", no=21, color="<blue>", icon=emoji.emojize(":rocket:"))
        self._safe_set_level("INFO", color="<blue>", icon=emoji.emojize(":placard:"))

    def _configure_output(self):
        self._log.remove()
        self._log.add(sys.stderr, format="{time:HH:mm:ss} | {level.icon} {level.name:<8} | {message}")

    def started(self, message, *args, **kwargs):
        self._log.log("STARTED", message, *args, **kwargs)

    def done(self, message, *args, **kwargs):
        self._log.log("DONE", message, *args, **kwargs)

    def progress(self, message, *args, **kwargs):
        self._log.log("PROGRESS", message, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._log, name)

# Type checker compatibility
if TYPE_CHECKING:
    log: CustomLogger
else:
    log = CustomLogger(pkg_logger)



