"""Custom exception classes for the AI-powered Structural MRI Analysis Platform."""


class BrainAIError(Exception):
    """Base exception for all platform errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ConfigurationError(BrainAIError):
    """Raised when environment or config parameters are invalid or missing."""


class MRIProcessingError(BrainAIError):
    """Raised when loading or preprocessing structural MRI scans fails."""


class ValidationFailedError(BrainAIError):
    """Raised when MRI data schemas fail validation checks."""


class ModelExecutionError(BrainAIError):
    """Raised when model inference, exporting, or loading fails."""


class ReportGenerationError(BrainAIError):
    """Raised when compiling the final PDF report fails."""
