"""Workflows package for integrated pipelines."""

from segmentvideo.workflows.segmentation import IntegratedSegmentationWorkflow, run_segmentation_workflow

__all__ = ['IntegratedSegmentationWorkflow', 'run_segmentation_workflow']
