class SageMakerShimError(Exception):
    """Base class for all exceptions"""


class ZipExtractionError(SageMakerShimError):
    """Raised when we could not extract a zip file"""
