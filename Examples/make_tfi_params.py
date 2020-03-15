import json
import sys

data = {
    "N" : 28,
    "alpha": 3,
    "h" : 1.0,
    "useCG": True,
    "Optimizer" :
        {
            "name": "SGD",
            "alpha": 0.02,
            "p": 0.0
        }
}
print(json.dumps(data, indent=4))
