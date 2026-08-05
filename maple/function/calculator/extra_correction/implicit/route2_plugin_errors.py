"""Shared fail-closed errors for the Route-2 MLIP plug-in boundary."""


class PluginContractError(TypeError):
    """A plug-in cannot enter a requested Route-2 scientific route."""


__all__ = ["PluginContractError"]
