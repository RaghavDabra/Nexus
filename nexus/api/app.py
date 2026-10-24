from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from nexus.agents.cost_agent import CostAgent
from nexus.agents.executor_agent import ExecutorAgent
from nexus.agents.health_agent import HealthAgent
from nexus.agents.router_agent import RouterAgent
from nexus.api.routes.agents import init_agent_routes, router as agents_router
from nexus.api.routes.analytics import init_analytics_routes, router as analytics_router
from nexus.api.routes.gateway import init_gateway_routes, router as gateway_router
from nexus.api.websocket import event_to_websocket, websocket_endpoint
from nexus.config import settings
from nexus.core.event_bus import EventBus
from nexus.core.memory import AgentMemory
from nexus.core.orchestrator import Orchestrator
from nexus.gateway.cost_tracker import CostTracker
from nexus.gateway.providers.mock_provider import MockProvider
from nexus.gateway.router import Router

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_system() -> tuple[Orchestrator, Router, CostTracker, HealthAgent]:
    event_bus = EventBus(redis_url=settings.redis_url)
    memory = AgentMemory(redis_url=settings.redis_url)

    model_router = Router()
    cost_tracker = CostTracker()

    providers = {}

    mock = MockProvider(failure_rate=0.05)
    model_router.register_provider(mock)
    providers["mock"] = mock

    if settings.openai_api_key:
        from nexus.gateway.providers.openai_provider import OpenAIProvider

        openai = OpenAIProvider(settings.openai_api_key)
        model_router.register_provider(openai)
        providers["openai"] = openai
        logger.info("OpenAI provider registered")

    if settings.anthropic_api_key:
        from nexus.gateway.providers.anthropic_provider import AnthropicProvider

        anthropic = AnthropicProvider(settings.anthropic_api_key)
        model_router.register_provider(anthropic)
        providers["anthropic"] = anthropic
        logger.info("Anthropic provider registered")

    orchestrator = Orchestrator(event_bus, memory)

    router_agent = RouterAgent(event_bus, memory, model_router)
    executor_agent = ExecutorAgent(event_bus, memory, providers)
    cost_agent = CostAgent(event_bus, memory, cost_tracker, hourly_budget=10.0)
    health_agent = HealthAgent(event_bus, memory, providers, check_interval=60.0)

    orchestrator.register_agent(router_agent)
    orchestrator.register_agent(executor_agent)
    orchestrator.register_agent(cost_agent)
    orchestrator.register_agent(health_agent)

    event_bus.subscribe("*", event_to_websocket)

    return orchestrator, model_router, cost_tracker, health_agent


_orchestrator: Orchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator
    orchestrator, model_router, cost_tracker, health_agent = build_system()
    _orchestrator = orchestrator

    init_gateway_routes(orchestrator, model_router)
    init_agent_routes(orchestrator)
    init_analytics_routes(cost_tracker, health_agent)

    await orchestrator.start()
    logger.info("Nexus AI Platform started")

    yield

    await orchestrator.stop()
    logger.info("Nexus AI Platform stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Nexus AI Platform",
        description="Multi-Agent AI Infrastructure Operations Platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(gateway_router)
    app.include_router(agents_router)
    app.include_router(analytics_router)
    app.add_websocket_route("/ws", websocket_endpoint)

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "agents": _orchestrator.agent_statuses() if _orchestrator else [],
        }

    return app


app = create_app()
