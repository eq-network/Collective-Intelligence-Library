import pandas as pd
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


@dataclass
class ExperimentResult:
    data: pd.DataFrame # Per-round data from one simulation run
    metadata: Dict[str, Any] # Metadata about the run (params, status, timing)

class ResultsAggregator:
    def __init__(self):
        self.all_run_data: List[pd.DataFrame] = []
        self.all_run_metadata: List[Dict[str, Any]] = []

    def add_result(self, df: pd.DataFrame, metadata: Dict[str, Any]):
        self.all_run_data.append(df)
        self.all_run_metadata.append(metadata)

    def get_concatenated_data(self) -> pd.DataFrame:
        if not self.all_run_data:
            return pd.DataFrame()
        return pd.concat(self.all_run_data, ignore_index=True)

    def get_metadata_summary(self) -> pd.DataFrame:
        if not self.all_run_metadata:
            return pd.DataFrame()
        return pd.DataFrame(self.all_run_metadata)

    def save_results(self, base_filename_prefix: str, timestamp: str):
        data_df = self.get_concatenated_data()
        metadata_df = self.get_metadata_summary()

        data_filename = f"{base_filename_prefix}_data_{timestamp}.csv.gz" # Use gzip for large data
        metadata_filename = f"{base_filename_prefix}_metadata_{timestamp}.csv"

        if not data_df.empty:
            data_df.to_csv(data_filename, index=False, compression='gzip')
            print(f"Aggregated run data saved to: {data_filename}")
        if not metadata_df.empty:
            metadata_df.to_csv(metadata_filename, index=False)
            print(f"Aggregated metadata saved to: {metadata_filename}")