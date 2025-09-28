import importlib.metadata

__version__ = '1.0'
# __version__ = importlib.metadata.version("mem0ai")

from mem0.client.main import AsyncMemoryClient, MemoryClient  # noqa
from mem0.memory.main import AsyncMemory, Memory  # noqa
