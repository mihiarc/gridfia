"""Heirs property analysis package."""

from .pipeline.executor import main
from .pipeline.stages import Pipeline, PipelineStage, PipelineConfig

__all__ = ['main', 'Pipeline', 'PipelineStage', 'PipelineConfig'] 