import pandas as pd
import json
import os

class AdversarialDeviationAnalyzer:
    def __init__(self, anomaly_logs_df: pd.DataFrame):
        self.df = anomaly_logs_df
        if self.df.empty:
            print("Warning: Anomaly logs DataFrame is empty.")

    def get_anomaly_summary(self):
        if self.df.empty:
            return {"message": "No anomalies to summarize."}
        return self.df['anomaly_type'].value_counts().to_dict()

    def sample_anomalies(self, anomaly_type_filter: str, n_samples=5, output_file=None, columns_to_show=None):
        if self.df.empty:
            print(f"No anomalies logged. Cannot sample for {anomaly_type_filter}.")
            return pd.DataFrame()

        subset = self.df[self.df['anomaly_type'] == anomaly_type_filter]
        if subset.empty:
            print(f"No '{anomaly_type_filter}' anomalies found.")
            return pd.DataFrame()
        
        samples_df = subset.sample(min(n_samples, len(subset)))
        
        if columns_to_show is None:
            columns_to_show = ['run_id', 'round_num', 'agent_id', 'mechanism', 
                               'raw_llm_prompt', 'raw_llm_response']
            if anomaly_type_filter == 'ADV_DELEGATES_TO_ALIGNED':
                columns_to_show.extend(['chosen_delegate_id', 'log_pld_delegate_performance_history_str'])
            elif anomaly_type_filter == 'ADV_VOTES_FOR_NON_WORST_EV':
                columns_to_show.extend(['approved_portfolio_idx', 'chosen_portfolio_perceived_ev', 'agent_perceived_evs_str'])
        
        for col in columns_to_show:
            if col not in samples_df.columns:
                samples_df[col] = pd.NA

        display_samples = samples_df[columns_to_show]

        print(f"\n--- Samples: {anomaly_type_filter} ---")
        for _, row in display_samples.iterrows():
            print(f"\nRun: {row.get('run_id', 'N/A')}, Round: {row.get('round_num', 'N/A')}, Agent: {row.get('agent_id', 'N/A')}")
            for col in columns_to_show:
                if col not in ['run_id', 'round_num', 'agent_id', 'mechanism']:
                    val = row.get(col, "N/A")
                    print_val = (str(val)[:200] + '...') if isinstance(val, str) and len(val) > 200 else val
                    print(f"  {col}: {print_val}")
            print("-" * 30)
        
        if output_file:
            full_sample_data_to_save = []
            for index in samples_df.index:
                full_sample_data_to_save.append(self.df.loc[index].to_dict())
            
            with open(output_file, 'w') as f:
                json.dump(full_sample_data_to_save, f, indent=2, default=str)
            print(f"Saved '{anomaly_type_filter}' samples to {output_file}")
        return samples_df

# Example usage
if __name__ == "__main__":
    # Update this path to your actual anomaly log file
    ANOMALY_LOG_CSV_PATH = r"experiment_outputs\TimelinePortfolioDemocracySuite_20250602_181944\aggregated_final_anomaly_logs_20250602_181944.csv\aggregated_final_anomaly_logs_20250602_181944.csv"
    
    if os.path.exists(ANOMALY_LOG_CSV_PATH):
        anomaly_df = pd.read_csv(ANOMALY_LOG_CSV_PATH)
        analyzer = AdversarialDeviationAnalyzer(anomaly_df)
        
        print("Anomaly Summary:")
        summary = analyzer.get_anomaly_summary()
        for anomaly_type, count in summary.items():
            print(f"  - {anomaly_type}: {count}")
        
        analyzer.sample_anomalies('ADV_DELEGATES_TO_ALIGNED', n_samples=5)
        analyzer.sample_anomalies('ADV_VOTES_FOR_NON_WORST_EV', n_samples=5)
    else:
        print(f"Anomaly log file not found: {ANOMALY_LOG_CSV_PATH}")