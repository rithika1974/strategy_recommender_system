"""Small framework-neutral route adapter for quantitative service methods."""

from __future__ import annotations

import re
from typing import Any

from src.schemas.analysis import AnalysisRequest, AnalysisValidationError
from src.services.quantitative import QuantitativeAnalysisService, UnknownStockError


ROUTES = (
    "GET /stocks",
    "GET /features/{stock}",
    "GET /market-context/{stock}",
    "GET /strategies/{stock}",
    "GET /backtest/{stock}",
    "GET /optimization/{stock}",
    "GET /robustness/{stock}",
)


class QuantitativeAPI:
    """Map endpoint-shaped requests to the reusable Python service facade."""

    _ENDPOINTS = {
        "features": "get_features",
        "market-context": "get_market_context",
        "strategies": "get_strategies",
        "backtest": "get_backtest",
        "optimization": "get_optimization",
        "robustness": "get_robustness",
    }

    def __init__(self, service: QuantitativeAnalysisService | None = None) -> None:
        self.service = service or QuantitativeAnalysisService()

    def handle(
        self,
        method: str,
        path: str,
        query: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Return an HTTP-compatible status code and JSON-compatible body."""
        if method.upper() != "GET":
            return 405, {"status": "error", "error": "method_not_allowed"}
        if path == "/stocks":
            return 200, self.service.list_stocks()
        match = re.fullmatch(
            r"/(features|market-context|strategies|backtest|optimization|robustness)/([^/]+)",
            path,
        )
        if not match:
            return 404, {"status": "error", "error": "route_not_found"}
        endpoint, stock = match.groups()
        try:
            request = AnalysisRequest(
                stock=stock,
                horizon=(query or {}).get("horizon"),
            )
            body = getattr(self.service, self._ENDPOINTS[endpoint])(request)
            return 200, body
        except AnalysisValidationError as exc:
            return 400, {
                "status": "error",
                "error": "invalid_analysis_request",
                "details": str(exc),
            }
        except UnknownStockError as exc:
            return 404, {"status": "error", "error": "unknown_stock", "details": str(exc)}
