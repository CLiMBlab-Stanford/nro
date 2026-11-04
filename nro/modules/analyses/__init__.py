"""Scheduled execution of study analyses.

Study analyses live in study repositories, not in this package. This package
plans their work items for the scheduler and runs one work item inside a
scheduled attempt. The analysis code itself runs through ``nro.analysis.runner``.
"""
