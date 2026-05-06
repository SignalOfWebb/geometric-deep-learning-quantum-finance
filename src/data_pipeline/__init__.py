"""Four-stage data pipeline that transforms raw financial CSVs into
model-ready Parquet features. Stages run sequentially: consolidate
raw files, merge asset classes, engineer features, then export.

References:
    McKinney, W. (2010). Data Structures for Statistical Computing
    in Python. Proceedings of the 9th Python in Science Conference.
"""

from .stage1_consolidate_equities import main as consolidate_equities
from .stage1_consolidate_macro import main as consolidate_macro
from .stage1_consolidate_commodities import main as consolidate_commodities
from .stage2_merge_master import main as merge_master
from .stage3_engineer_features import main as engineer_features
from .stage4_export_parquet import main as export_parquet

__all__ = [
    "consolidate_equities",
    "consolidate_macro",
    "consolidate_commodities",
    "merge_master",
    "engineer_features",
    "export_parquet",
]
