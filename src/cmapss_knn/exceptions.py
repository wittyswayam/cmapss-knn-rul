"""Project-specific exceptions, so callers can distinguish failure modes."""

from __future__ import annotations


class CMAPSSError(Exception):
    """Base class for all errors raised by this package."""


class ConfigError(CMAPSSError):
    """The configuration file is missing, malformed or incomplete."""


class DataNotFoundError(CMAPSSError):
    """A required raw data file or archive could not be found."""


class DataFormatError(CMAPSSError):
    """A raw data file exists but cannot be parsed into the expected layout."""


class DataValidationError(CMAPSSError):
    """Parsed data violates one or more integrity checks."""


class ModelNotFittedError(CMAPSSError, RuntimeError):
    """``predict`` or ``kneighbors`` was called before ``fit``."""
