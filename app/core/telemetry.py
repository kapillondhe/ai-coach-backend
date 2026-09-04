import os
from typing import cast

from fastapi import FastAPI
from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from phoenix.otel import TracerProvider as PhoenixTracerProvider
from phoenix.otel import register
from pydantic_ai import Agent

from app.core.config import get_settings


def setup_telemetry(app: FastAPI) -> None:
    """Wire up tracing to Arize Phoenix. No-ops if PHOENIX_API_KEY is unset, so local dev
    and tests behave exactly as they do without Phoenix configured."""
    settings = get_settings()
    if not settings.phoenix_api_key:
        return

    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = settings.phoenix_collector_endpoint

    # register() is typed as returning the base opentelemetry TracerProvider, but it always
    # constructs and returns phoenix.otel.TracerProvider, whose add_span_processor() accepts
    # replace_default_processor (needed below to keep register()'s exporting processor alive).
    tracer_provider = cast(
        PhoenixTracerProvider,
        register(
            project_name=settings.phoenix_project_name,
            api_key=settings.phoenix_api_key,
            resource=Resource.create({"service.name": settings.otel_service_name}),
            batch=True,
        ),
    )
    tracer_provider.add_span_processor(OpenInferenceSpanProcessor(), replace_default_processor=False)

    Agent.instrument_all()
    FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)
