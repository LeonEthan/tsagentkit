"""Adapter registry for TSFM discovery and factory creation.

Provides centralized registration and discovery of TSFM adapters
with availability checking and dependency management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import AdapterConfig, TSFMAdapter


class AdapterRegistry:
    """Registry for TSFM adapters with auto-discovery.

    Provides centralized access to available adapters with
    fallback handling and version checking.

    Example:
        >>> # Register an adapter
        >>> AdapterRegistry.register("chronos", ChronosAdapter)
        >>>
        >>> # List available adapters
        >>> AdapterRegistry.list_available()
        ['chronos', 'moirai']
        >>>
        >>> # Check if adapter can be used
        >>> AdapterRegistry.check_availability("chronos")
        (True, None)
        >>>
        >>> # Create adapter instance
        >>> adapter = AdapterRegistry.create("chronos", config)
    """

    _adapters: dict[str, type[TSFMAdapter]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        adapter_class: type[TSFMAdapter],
    ) -> None:
        """Register an adapter class.

        Args:
            name: Unique identifier for the adapter
            adapter_class: The adapter class to register

        Raises:
            ValueError: If name is already registered with a different class
        """
        if name in cls._adapters and cls._adapters[name] is not adapter_class:
            raise ValueError(
                f"Adapter '{name}' is already registered with a different class"
            )
        cls._adapters[name] = adapter_class

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister an adapter.

        Args:
            name: Name of the adapter to unregister

        Raises:
            KeyError: If adapter is not registered
        """
        if name not in cls._adapters:
            raise KeyError(f"Adapter '{name}' is not registered")
        del cls._adapters[name]

    @classmethod
    def get(cls, name: str) -> type[TSFMAdapter]:
        """Get adapter class by name.

        Args:
            name: Adapter name

        Returns:
            The adapter class

        Raises:
            ValueError: If adapter is not registered
        """
        if name not in cls._adapters:
            available = ", ".join(sorted(cls._adapters.keys()))
            raise ValueError(
                f"Unknown adapter '{name}'. "
                f"Available adapters: {available or 'none'}"
            )
        return cls._adapters[name]

    @classmethod
    def list_available(cls) -> list[str]:
        """List all registered adapter names.

        Returns:
            Sorted list of registered adapter names
        """
        return sorted(cls._adapters.keys())

    @classmethod
    def create(
        cls,
        name: str,
        config: AdapterConfig | None = None,
    ) -> TSFMAdapter:
        """Factory method to create adapter instance.

        Args:
            name: Name of the adapter to create
            config: Optional configuration (uses defaults if None)

        Returns:
            Instantiated adapter

        Raises:
            ValueError: If adapter is not registered
        """
        from .base import AdapterConfig

        adapter_class = cls.get(name)
        return adapter_class(config or AdapterConfig(model_name=name))

    @classmethod
    def check_availability(cls, name: str) -> tuple[bool, str]:
        """Check if adapter dependencies are installed.

        Args:
            name: Adapter name to check

        Returns:
            Tuple of (is_available, error_message). error_message is empty string
            if the adapter is available.
        """
        try:
            adapter_class = cls.get(name)
        except ValueError as e:
            return False, str(e)

        try:
            adapter_class._check_dependencies()
            return True, ""
        except ImportError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error checking dependencies: {e}"

    @classmethod
    def get_available_adapters(cls) -> dict[str, type[TSFMAdapter]]:
        """Get all adapters that have their dependencies installed.

        Returns:
            Dictionary mapping available adapter names to their classes
        """
        available = {}
        for name in cls._adapters:
            is_avail, _ = cls.check_availability(name)
            if is_avail:
                available[name] = cls._adapters[name]
        return available

    @classmethod
    def clear(cls) -> None:
        """Clear all registered adapters.

        Useful for testing.
        """
        cls._adapters.clear()
