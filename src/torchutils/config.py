
import os
import dataclasses

@dataclasses.dataclass
class Config:
    debug: bool = False
    
    
config = Config(debug=os.environ.get("DEBUG", "0") == "1")