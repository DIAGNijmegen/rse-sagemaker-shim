class UserSafeError(Exception):
    """Messages are returned to the user"""


class InferenceTimeoutError(UserSafeError):
    """Time limit exceeded for inference"""
