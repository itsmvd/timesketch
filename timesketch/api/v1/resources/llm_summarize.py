# llm_summarize.py (Modified to match original error handling)

import logging
import json
import time
import pandas as pd

from typing import Optional

from flask import request, abort, jsonify, current_app
from flask_login import login_required, current_user
from flask_restful import Resource

from timesketch.api.v1 import resources, export
from timesketch.lib import definitions, utils
from timesketch.lib.llms.client import LLMClientService
from timesketch.models.sketch import Sketch

logger = logging.getLogger("timesketch.api.llm_summarize")

summary_response_schema = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}


class LLMSummarizeResource(resources.ResourceMixin, Resource):
    """Resource to get LLM summary of events."""

    llm_service = LLMClientService()
    def _get_prompt_text(self, events_dict: list) -> str:
        """Reads the prompt template from file and injects events."""
        prompt_file_path = current_app.config.get("PROMPT_LLM_SUMMARIZATION")
        if not prompt_file_path:
            logger.error("PROMPT_LLM_SUMMARIZATION config not set")
            abort(
                definitions.HTTP_STATUS_CODE_INTERNAL_SERVER_ERROR,
                "LLM summarization prompt path not configured.",
            )

        try:
            with open(prompt_file_path, "r", encoding="utf-8") as file_handle:
                prompt_template = file_handle.read()
        except FileNotFoundError:
            logger.error("Prompt file not found: %s", prompt_file_path)
            abort(
                definitions.HTTP_STATUS_CODE_INTERNAL_SERVER_ERROR,
                "LLM Prompt file not found on the server.",
            )
        except IOError as e:
            logger.error("Error reading prompt file: %s", e)
            abort(
                definitions.HTTP_STATUS_CODE_INTERNAL_SERVER_ERROR,
                "Error reading LLM prompt file.",
            )

        prompt_text = self.llm_service.prompt_from_template(
            "llm_summarization",
            prompt_template,
            {"events_json": json.dumps(events_dict)},
        )
        return prompt_text

    @login_required
    def post(self, sketch_id: int):
        """Handles POST request to the resource."""
        start_time = time.time()

        sketch = Sketch.get_with_acl(sketch_id)
        if not sketch:
            abort(
                definitions.HTTP_STATUS_CODE_NOT_FOUND,
                "No sketch found with this ID.",
            )
        if not sketch.has_permission(current_user, "read"):
            abort(
                definitions.HTTP_STATUS_CODE_FORBIDDEN,
                "User does not have read access controls on sketch.",
            )

        form = request.json
        if not form:
            abort(
                definitions.HTTP_STATUS_CODE_BAD_REQUEST,
                "The POST request requires data",
            )

        query_filter = form.get("filter", {})
        query_string = form.get("query", "*")
        if not query_string:
            query_string = "*"

        events_df = self._run_timesketch_query(sketch, query_string, query_filter)

        if events_df is None or events_df.empty:
            return jsonify(
                {"summary": "No events to summarize based on the current filter."}
            )
        new_df = events_df[["message"]]
        unique_df = new_df.drop_duplicates(subset="message", keep="first")
        events_dict = unique_df.to_dict(orient="records")

        total_events_count = len(new_df)
        unique_events_count = len(unique_df)

        # Use the service's metrics
        self.llm_service.metrics["llm_requests_total"].labels(
            feature_name="llm_summarization"
        ).inc()
        self.llm_service.metrics["llm_summary_events_processed_total"].labels(
            sketch_id=str(sketch_id)
        ).inc(total_events_count)
        self.llm_service.metrics["llm_summary_unique_events_total"].labels(
            sketch_id=str(sketch_id)
        ).inc(unique_events_count)

        logger.debug("Summarizing %d events", total_events_count)
        logger.debug("Reduced to %d unique events", unique_events_count)

        if not events_dict:
            return jsonify(
                {"summary": "No events to summarize based on the current filter."}
            )

        prompt_text = self._get_prompt_text(events_dict)

        try:
            response = self.llm_service.generate_with_timeout(
                feature_name="llm_summarization",
                prompt=prompt_text,
                response_schema=summary_response_schema,
            )
            #  NO ERROR TYPE CHECK HERE

        except TimeoutError:
            # Use service's metrics
            self.llm_service.metrics["llm_errors_total"].labels(
                feature_name="llm_summarization", error_type="timeout"
            ).inc()
            abort(definitions.HTTP_STATUS_CODE_BAD_REQUEST, "LLM call timed out.")
        except Exception as e:
            logger.error("Unable to call LLM [...] Error: %s", e)
            self.llm_service.metrics["llm_errors_total"].labels(
                feature_name="llm_summarization",
                error_type="llm_api_error",
            ).inc()
            abort(definitions.HTTP_STATUS_CODE_BAD_REQUEST, "Unable to get LLM data")

        if (
            not response
            or not response.get("response")
            or not response.get("response", {}).get("summary")
        ):
            logger.error("No valid summary from LLM.")
            self.llm_service.metrics["llm_errors_total"].labels(
                feature_name="llm_summarization",
                error_type="no_summary_error",
            ).inc()
            abort(
                definitions.HTTP_STATUS_CODE_BAD_REQUEST,
                "No valid summary from LLM.",
            )

        summary_text = response.get("response", {}).get("summary")

        duration = time.time() - start_time
        self.llm_service.metrics["llm_duration_seconds"].labels(
            feature_name="llm_summarization"
        ).observe(duration)

        return jsonify(
            {
                "summary": summary_text,
                "summary_event_count": total_events_count,
                "summary_unique_event_count": unique_events_count,
            }
        )

    def _run_timesketch_query(
        self,
        sketch: Sketch,
        query_string: str = "*",
        query_filter: Optional[dict] = None,
        id_list: Optional[list] = None,
    ) -> pd.DataFrame:
        """Runs a Timesketch query."""
        if not query_filter:
            query_filter = {}

        if id_list:
            id_query = " OR ".join([f'_id:"{event_id}"' for event_id in id_list])
            query_string = id_query

        all_indices = list({t.searchindex.index_name for t in sketch.timelines})
        indices = query_filter.get("indices", all_indices)

        if "_all" in indices:
            indices = all_indices

        indices, timeline_ids = utils.get_validated_indices(indices, sketch)

        if not indices:
            abort(
                definitions.HTTP_STATUS_CODE_BAD_REQUEST,
                "No valid search indices were found to perform the search on.",
            )

        result = self.datastore.search(
            sketch_id=sketch.id,
            query_string=query_string,
            query_filter=query_filter,
            query_dsl="",
            indices=indices,
            timeline_ids=timeline_ids,
        )

        return export.query_results_to_dataframe(result, sketch)
