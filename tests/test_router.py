from nexus.gateway.providers.base import CompletionRequest
from nexus.gateway.providers.mock_provider import MockProvider
from nexus.gateway.router import Router, TaskClassifier


def test_task_classifier_code():
    c = TaskClassifier()
    assert c.classify([{"role": "user", "content": "Write a Python function to sort a list"}]) == "code"


def test_task_classifier_reasoning():
    c = TaskClassifier()
    assert c.classify([{"role": "user", "content": "Analyze why this approach is better"}]) == "reasoning"


def test_task_classifier_extraction():
    c = TaskClassifier()
    assert c.classify([{"role": "user", "content": "Extract the key data points from this text"}]) == "extraction"


def test_router_routes_to_available_model():
    router = Router()
    mock = MockProvider()
    router.register_provider(mock)

    request = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
    decision = router.route(request)

    assert decision.provider == "mock"
    assert decision.model in ("mock-fast", "mock-quality")
    assert decision.confidence > 0


def test_router_cost_strategy_prefers_cheap():
    router = Router()
    mock = MockProvider()
    router.register_provider(mock)

    request = CompletionRequest(messages=[{"role": "user", "content": "Quick question"}])
    decision = router.route(request, strategy="cost")

    assert decision.provider == "mock"


def test_router_skips_unhealthy_providers():
    router = Router()
    mock = MockProvider()
    router.register_provider(mock)

    request = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
    health = {"mock:mock-fast": False, "mock:mock-quality": True}
    decision = router.route(request, health_status=health)

    assert decision.model == "mock-quality"


def test_prompt_hash_deterministic():
    msgs = [{"role": "user", "content": "test"}]
    assert Router.hash_prompt(msgs) == Router.hash_prompt(msgs)
