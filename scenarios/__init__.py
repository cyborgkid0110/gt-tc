"""Scenario package: persistence, region geometry, and coverage deployment.

Public API re-exported here so callers can simply::

    from scenarios import save_scenario, load_scenario

Submodules:
  - store               save_scenario / load_scenario (CSV persistence)
  - regions             shapely Region geometry for coverage scenarios
  - coverage_deploy     projected-PSO placement + connectivity repair
  - freeze_scenarios    snapshot the random deployments to scenarios/gen/*.csv
                        (run: python -m scenarios.freeze_scenarios)
  - make_coverage_scenario   region-def YAML -> coverage CSV
                        (run: python -m scenarios.make_coverage_scenario --def ...)

Data directories:
  - defs/   region-definition YAMLs
  - gen/    generated scenario CSVs
"""
from .store import save_scenario, load_scenario

__all__ = ['save_scenario', 'load_scenario']
