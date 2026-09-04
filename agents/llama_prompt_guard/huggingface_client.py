"""HuggingFace Inference API client for Llama Prompt Guard 2.

This module provides a client for the Llama Prompt Guard 2 model
via the HuggingFace Inference API for detecting prompt injection attacks.
"""

import os
from typing import NamedTuple

from huggingface_hub import InferenceClient


class PromptGuardResult(NamedTuple):
    """Result from Llama Prompt Guard 2 classification."""
    is_malicious: bool
    label: str
    score: float


class LlamaPromptGuardClient:
    """Client for Llama Prompt Guard 2 via HuggingFace Inference API.

    This client uses the HuggingFace Inference API to classify text
    as BENIGN or MALICIOUS for prompt injection detection.

    Attributes:
        MODEL_ID: The HuggingFace model ID for Llama Prompt Guard 2 86M.
    """

    MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"

    def __init__(self, api_token: str | None = None, threshold: float = 0.5):
        """Initialize the Llama Prompt Guard client.

        Args:
            api_token: HuggingFace API token. If not provided, uses
                      HF_API_TOKEN environment variable.
            threshold: Confidence threshold for classifying as malicious.
                      Default is 0.5.
        """
        self.api_token = api_token or os.getenv("HF_API_TOKEN")
        if not self.api_token:
            raise ValueError(
                "HuggingFace API token is required. "
                "Provide api_token or set HF_API_TOKEN environment variable."
            )
        self.threshold = threshold
        self.client = InferenceClient(token=self.api_token)

    # Label mapping: HuggingFace Inference API returns LABEL_0/LABEL_1
    # From the model: LABEL_0 = Benign, LABEL_1 = Malicious
    LABEL_MAP = {
        "LABEL_0": "BENIGN",
        "LABEL_1": "MALICIOUS",
        "BENIGN": "BENIGN",
        "MALICIOUS": "MALICIOUS",
    }

    def classify(self, text: str) -> PromptGuardResult:
        """Classify text for prompt injection.

        Args:
            text: The text to classify.

        Returns:
            PromptGuardResult with is_malicious flag, label, and confidence score.
        """
        # HuggingFace text_classification returns list[TextClassificationOutputElement]
        # Each element has .label and .score attributes
        results = self.client.text_classification(text, model=self.MODEL_ID)

        # Find the MALICIOUS score
        # Labels may be LABEL_0/LABEL_1 or BENIGN/MALICIOUS depending on API version
        malicious_score = 0.0
        benign_score = 0.0
        for result in results:
            mapped_label = self.LABEL_MAP.get(result.label, result.label)
            if mapped_label == "MALICIOUS":
                malicious_score = result.score
            elif mapped_label == "BENIGN":
                benign_score = result.score

        # Determine if malicious based on threshold
        is_malicious = malicious_score >= self.threshold

        return PromptGuardResult(
            is_malicious=is_malicious,
            label="MALICIOUS" if is_malicious else "BENIGN",
            score=malicious_score if is_malicious else benign_score
        )
