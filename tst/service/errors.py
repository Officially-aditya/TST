"""Errors exposed by the application service boundary."""


class ServiceError(RuntimeError):
    """Base class for errors that can be translated by CLI or HTTP clients."""


class ProjectNotFoundError(ServiceError):
    pass


class MemoryNotFoundError(ServiceError):
    pass


class ScopeMovementError(ServiceError):
    pass


class IntegrationError(ServiceError):
    pass
