import os
import dataclasses




@dataclasses.dataclass
class BenchConfig:
    EXPORT_TRACE: bool = False
    
bench_config = BenchConfig()
bench_config.EXPORT_TRACE = os.environ.get("BENCH_EXPORT_TRACE", "0") == "1"


@dataclasses.dataclass
class Config:
    debug: bool = False


config = Config(debug=os.environ.get("DEBUG", "0") == "1")

