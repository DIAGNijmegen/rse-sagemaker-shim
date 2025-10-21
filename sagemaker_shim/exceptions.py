class UserSafeError(Exception):
    """Messages are returned to the user"""


class ZipExtractionError(UserSafeError):
    """Raised when we could not extract a zip file"""
