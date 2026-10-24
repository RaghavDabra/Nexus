from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from nexus.agents.cost_agent import CostAgent
from nexus.agents.executor_agent import ExecutorAgent
from nexus.agents.gpu_monitor_agent import GPUMonitorAgent
from nexus.agents.gpu_scheduler_agent import GPUSchedulerAgent
from nexus.agents.health_agent import HealthAgent
from nexus.agents.router_agent import RouterAgent
from nexus.api.routes.agents import init_agent_routes, router as agents_router
from nexus.api.routes.analytics import init_analytics_routes, router as analytics_router
from nexus.api.routes.gateway import init_gateway_routes, router as gateway_router
from nexus.api.routes.gpu import init_gpu_routes, router as gpu_router
from nexus.api.websocket import event_to_websocket, websocket_endpoint
from nexus.config import settings
from nexus.core.event_bus import EventBus
from nexus.core.memory import AgentMemory
from nexus.core.orchestrator import Orchestrator
from nexus.gateway.cost_tracker import CostTracker
from nexus.gateway.providers.mock_provider import MockProvider
from nexus.gateway.router import Router
from nexus.gpu.metrics import MockMetricsCollector
from nexus.gpu.predictor import WorkloadPredictor
from nexus.gpu.scheduler import GPUScheduler

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class SystemComponents:
    orchestrator: Orchestrator
    model_router: Router
    cost_tracker: CostTracker
    health_agent: HealthAgent
    gpu_collector: MockMetricsCollector
    gpu_scheduler: GPUScheduler
    gpu_predictor: WorkloadPredictor


def build_system() -> SystemComponents:
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

    gpu_collector = MockMetricsCollector(num_gpus=8)
    gpu_scheduler = GPUScheduler()
    gpu_predictor = WorkloadPredictor()

    gpu_monitor = GPUMonitorAgent(event_bus, memory, gpu_collector, gpu_predictor, poll_interval=15.0)
    gpu_sched_agent = GPUSchedulerAgent(event_bus, memory, gpu_scheduler, gpu_collector, gpu_predictor)

    orchestrator.register_agent(gpu_monitor)
    orchestrator.register_agent(gpu_sched_agent)

    event_bus.subscribe("*", event_to_websocket)

    return SystemComponents(
        orchestrator=orchestrator,
        model_router=model_router,
        cost_tracker=cost_tracker,
        health_agent=health_agent,
        gpu_collector=gpu_collector,
        gpu_scheduler=gpu_scheduler,
        gpu_predictor=gpu_predictor,
    )


_orchestrator: Orchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator
    system = build_system()
    _orchestrator = system.orchestrator

    init_gateway_routes(system.orchestrator, system.model_router)
    init_agent_routes(system.orchestrator)
    init_analytics_routes(system.cost_tracker, system.health_agent)
    init_gpu_routes(system.gpu_collector, system.gpu_scheduler, system.gpu_predictor, None)

    await system.orchestrator.start()
    logger.info("Nexus AI Platform started with %d agents", len(system.orchestrator.agents))

    yield

    await system.orchestrator.stop()
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
    app.include_router(gpu_router)
    app.add_websocket_route("/ws", websocket_endpoint)

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "agents": _orchestrator.agent_statuses() if _orchestrator else [],
        }

    return app


app = create_app()
