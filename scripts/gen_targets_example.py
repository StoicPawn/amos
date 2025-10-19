"""Generate an example targets.csv file."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def main(path: str = "data/targets/targets.csv") -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"symbol": ["AAPL", "MSFT", "NVDA"], "target_weight": [0.03, 0.02, -0.01]})
    tmp_path = target_path.with_suffix(".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(target_path)


if __name__ == "__main__":
    main()
