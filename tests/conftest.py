"""Shared test configuration."""
import pytest

pytest_plugins = []


def pytest_configure(config):
    """Register custom marks."""
    config.addinimark("live_local", doc="Requires live Ollama to be running.")
