import asyncio

import pytest

from nexus.core.event_bus import Event, EventBus
from nexus.core.memory import AgentMemory


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def memory():
    return AgentMemory()


async def test_event_bus_local_pubsub(event_bus):
    received = []

    async def handler(event: Event):
        received.append(event)

    await event_bus.connect()
    event_bus.subscribe("test.event", handler)

    await event_bus.publish(Event(type="test.event", payload={"data": 1}, source="test"))
    await asyncio.sleep(0.01)

    assert len(received) == 1
    assert received[0].payload["data"] == 1
    await event_bus.disconnect()


async def test_event_bus_wildcard(event_bus):
    received = []

    async def handler(event: Event):
        received.append(event)

    await event_bus.connect()
    event_bus.subscribe("*", handler)

    await event_bus.publish(Event(type="any.event", payload={}, source="test"))
    await asyncio.sleep(0.01)

    assert len(received) == 1
    await event_bus.disconnect()


async def test_memory_store_and_recall(memory):
    await memory.connect()
    await memory.store("agent-1", "key1", {"value": 42})

    result = await memory.recall("agent-1", "key1")
    assert result == {"value": 42}

    missing = await memory.recall("agent-1", "nonexistent")
    assert missing is None
    await memory.disconnect()


async def test_memory_shared_namespace(memory):
    await memory.connect()
    await memory.share("shared_key", "shared_value")

    result = await memory.recall_shared("shared_key")
    assert result == "shared_value"
    await memory.disconnect()


async def test_memory_delete(memory):
    await memory.connect()
    await memory.store("agent-1", "temp", "data")
    await memory.delete("agent-1", "temp")

    result = await memory.recall("agent-1", "temp")
    assert result is None
    await memory.disconnect()
