from enum import Enum


class DatasetFileType(str, Enum):
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"


class DatasetJoinKind(str, Enum):
    INNER = "inner"
    LEFT_OUTER = "left_outer"
    OUTER = "outer"
    RIGHT_OUTER = "right_outer"
