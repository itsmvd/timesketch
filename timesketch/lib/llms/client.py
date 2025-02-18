# timesketch/lib/llms/client.py
import multiprocessing
import multiprocessing.managers
import logging
import time
from typing import Optional, Dict, Any

from flask import current_app
import prometheus_client

from timesketch.lib.llms.manager import LLMManager
from timesketch.lib.definitions import METRICS_NAMESPACE

logger = logging.getLogger("timesketch.llm_client")

DEFAULT_TIMEOUT = 30
LLM_METRICS_REGISTRY = prometheus_client.CollectorRegistry()


class LLMClientService:
    """Service for interacting with LLM providers."""

    _metrics = None

    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        """Initialize the LLM client service.

        Args:
            timeout: Default timeout in seconds for LLM calls.
        """
        self.timeout = timeout
        if LLMClientService._metrics is None:  # Only define metrics once
            LLMClientService._metrics = self._define_metrics()

    @property
    def metrics(self):
        return LLMClientService._metrics

    def _define_metrics(self):
        """Defines Prometheus metrics for LLM interactions."""
        metrics = {
            "llm_requests_total": prometheus_client.Counter(
                "llm_requests_total",
                "Total number of LLM requests",
                ["feature_name"],
                namespace=METRICS_NAMESPACE,
                registry=LLM_METRICS_REGISTRY,
            ),
            "llm_errors_total": prometheus_client.Counter(
                "llm_errors_total",
                "Total number of LLM errors",
                ["feature_name", "error_type"],
                namespace=METRICS_NAMESPACE,
                registry=LLM_METRICS_REGISTRY,
            ),
            "llm_duration_seconds": prometheus_client.Summary(
                "llm_duration_seconds",
                "Time taken for LLM requests (in seconds)",
                ["feature_name"],
                namespace=METRICS_NAMESPACE,
                registry=LLM_METRICS_REGISTRY,
            ),
            "llm_timed_out_total": prometheus_client.Counter(
                "llm_timed_out_total",
                "Total number of LLM requests that timed out",
                ["feature_name"],
                namespace=METRICS_NAMESPACE,
                registry=LLM_METRICS_REGISTRY,
            ),
            # Summarization-specific metrics:
            "llm_summary_events_processed_total": prometheus_client.Counter(
                "llm_summary_events_processed_total",
                "Total number of events processed for LLM summarization",
                ["sketch_id"],  # Keep sketch_id label
                namespace=METRICS_NAMESPACE,
                registry=LLM_METRICS_REGISTRY,
            ),
            "llm_summary_unique_events_total": prometheus_client.Counter(
                "llm_summary_unique_events_total",
                "Total number of unique events sent to the LLM for summarization",
                ["sketch_id"],
                namespace=METRICS_NAMESPACE,
                registry=LLM_METRICS_REGISTRY,
            ),
        }
        return metrics

    def _get_provider(self, feature_name: str):
        """Gets an LLM provider instance for the given feature."""
        try:
            return LLMManager.create_provider(feature_name=feature_name)
        except Exception as e:
            logger.error("Error creating LLM provider: %s", e, exc_info=True)
            self.metrics["llm_errors_total"].labels(
                feature_name=feature_name, error_type="provider_creation"
            ).inc()
            raise ValueError(
                f"Unable to create LLM provider for feature '{feature_name}'"
            ) from e

    def generate_with_timeout(
        self,
        feature_name: str,
        prompt: str,
        response_schema: Optional[dict] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generates a response from an LLM with a timeout."""
        start_time = time.monotonic()
        self.metrics["llm_requests_total"].labels(feature_name=feature_name).inc()

        effective_timeout = timeout or self.timeout

        try:
            with multiprocessing.Manager() as manager:
                shared_response = manager.dict()
                process = multiprocessing.Process(
                    target=self._generate_in_process,
                    args=(
                        feature_name,
                        prompt,
                        response_schema,
                        shared_response,
                    ),
                )
                process.start()
                process.join(timeout=effective_timeout)

                if process.is_alive():
                    process.terminate()
                    process.join()
                    self.metrics["llm_timed_out_total"].labels(
                        feature_name=feature_name
                    ).inc()
                    self.metrics["llm_errors_total"].labels(
                        feature_name=feature_name, error_type="timeout"
                    ).inc()
                    raise TimeoutError(
                        f"LLM call for {feature_name} timed out after"
                        f" {effective_timeout} seconds"
                    )

                response = dict(shared_response)

        except TimeoutError:
            raise
        except Exception as e:
            self.metrics["llm_errors_total"].labels(
                feature_name=feature_name, error_type="generation_error"
            ).inc()
            raise ValueError(
                f"Error generating LLM response for {feature_name}: {e}"
            ) from e
        finally:
            duration = time.monotonic() - start_time
            self.metrics["llm_duration_seconds"].labels(
                feature_name=feature_name
            ).observe(duration)

        return response

    def _generate_in_process(
        self,
        feature_name: str,
        prompt: str,
        response_schema: Optional[dict],
        shared_response: multiprocessing.managers.DictProxy,
    ):
        """Generates an LLM response within a separate process."""
        try:
            llm_provider = self._get_provider(feature_name)
            prediction = llm_provider.generate(prompt, response_schema)
            shared_response.update({"response": prediction})

        except Exception as e:
            logger.error("Error in LLM generation process: %s", e, exc_info=True)
            shared_response.update({"error": str(e), "feature_name": feature_name})

    def prompt_from_template(
        self, feature_name: str, template: str, kwargs: dict
    ) -> str:
        """
        Formats a prompt from a template, using the same logic as the LLMProvider class.
        Potentially, an LLM provider could provide different formatters in the future,
        that's why we pass in the feature_name.

        Args:
            feature_name: The name of the feature using the LLM.
            template: The template string.
            kwargs: A dictionary of keyword arguments to substitute in the template.
        """

        llm_provider = self._get_provider(feature_name)
        return llm_provider.prompt_from_template(template, kwargs)
