"""Modular Multi-Expert Retrieval (MMER).

Independent retrieval experts + a learned fusion head. See ``docs/retrieval/05_modular_design.md``.
"""

from gsr_cacl.experts.base import Expert, minmax
from gsr_cacl.experts.lexical import LexicalExpert
from gsr_cacl.experts.concept import ConceptExpert
from gsr_cacl.experts.cell import CellExpert
from gsr_cacl.experts.entity import EntityExpert

__all__ = ["Expert", "minmax", "LexicalExpert", "ConceptExpert", "CellExpert", "EntityExpert"]
