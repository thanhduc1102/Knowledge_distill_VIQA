"""Generation phase: fact-grounded generator + deterministic verifier + Number-Match."""

from gsr_cacl.generation.generator import (
    BaseGenerator,
    ExtractiveGenerator,
    HFGenerator,
    build_generator,
)
from gsr_cacl.generation.verifier import (
    VerificationResult,
    verify,
    is_grounded,
    is_derivable,
    extract_final_number,
)
from gsr_cacl.generation.metrics import compute_number_match
from gsr_cacl.generation.prompts import build_chat_messages, build_user_prompt, SYSTEM_PROMPT

__all__ = [
    "BaseGenerator", "ExtractiveGenerator", "HFGenerator", "build_generator",
    "VerificationResult", "verify", "is_grounded", "is_derivable", "extract_final_number",
    "compute_number_match",
    "build_chat_messages", "build_user_prompt", "SYSTEM_PROMPT",
]
