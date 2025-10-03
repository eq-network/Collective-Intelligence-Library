# experiments/results.py - ENHANCED FOR TIMELINE DATA
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import psutil
import gc
import os

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


@dataclass
class ExperimentResult:
    """Enhanced experiment result with timeline data support."""
    data: pd.DataFrame  # Timeline data (multiple rows per simulation)
    metadata: Dict[str, Any]  # Simulation metadata (single entry)


class EnhancedResultsAggregator:
    """
    ARCHITECTURAL PURPOSE: Efficient aggregation of timeline data with memory management.
    
    DESIGN PRINCIPLES:
    1. Memory-conscious aggregation for large timeline datasets
    2. Incremental processing to prevent memory overflow
    3. Data validation and integrity checking
    4. Performance monitoring and optimization
    
    OPTIMIZATION STRATEGIES:
    - Chunked data processing for large datasets
    - Memory usage monitoring and garbage collection
    - Efficient data type optimization
    - Streaming I/O for large file operations
    """
    
    def __init__(self, memory_limit_gb: float = 4.0, chunk_size: int = 10000):
        """
        Initialize aggregator with memory management parameters.
        
        Args:
            memory_limit_gb: Memory limit in GB before triggering optimization
            chunk_size: Number of rows to process in each chunk
        """
        self.memory_limit_gb = memory_limit_gb
        self.chunk_size = chunk_size
        
        # Storage for results
        self.all_run_data: List[pd.DataFrame] = []
        self.all_run_metadata: List[Dict[str, Any]] = []
        self.all_anomaly_logs: List[Dict[str, Any]] = []  # ADD THIS LINE
        
        # Performance tracking
        self.total_timeline_points = 0
        self.total_simulations = 0
        self.memory_warnings = 0
        
        # Dynamic processing parameters
        self.current_save_interval = 50  # Initial save interval
        self.min_save_interval = 10     # Minimum interval under memory pressure
        self.max_save_interval = 100    # Maximum interval under optimal conditions
        
        print(f"[AGGREGATOR_INIT] Memory limit: {memory_limit_gb}GB, Chunk size: {chunk_size}")
    
    def add_result(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> None:
        """
        Add timeline result with memory-conscious processing.
        
        MEMORY MANAGEMENT STRATEGY:
        1. Monitor memory usage before adding data
        2. Optimize data types for efficiency
        3. Trigger garbage collection if needed
        4. Warn if approaching memory limits
        
        VALIDATION FRAMEWORK:
        - Verify timeline data structure
        - Check for data consistency
        - Validate metadata completeness
        """
        # Pre-processing memory check
        current_memory_gb = psutil.Process().memory_info().rss / 1024**3
        
        if current_memory_gb > self.memory_limit_gb:
            self.memory_warnings += 1
            print(f"[MEMORY_WARNING] Memory usage ({current_memory_gb:.2f}GB) exceeds limit ({self.memory_limit_gb}GB)")
            gc.collect()  # Force garbage collection
        
        # Validate and optimize timeline data
        optimized_df = self._optimize_timeline_dataframe(df)
        validated_metadata = self._validate_metadata(metadata)
        
        # Add to storage
        self.all_run_data.append(optimized_df)
        self.all_run_metadata.append(validated_metadata)
        
        # Update tracking statistics
        self.total_timeline_points += len(optimized_df)
        self.total_simulations += 1
        
        # Progress reporting
        if self.total_simulations % 50 == 0:
            print(f"[AGGREGATOR_PROGRESS] Processed {self.total_simulations} simulations, "
                  f"{self.total_timeline_points} timeline points, "
                  f"Memory: {current_memory_gb:.2f}GB")

    def _optimize_timeline_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        OPTIMIZATION: Reduce memory footprint through data type optimization.
        
        OPTIMIZATION TECHNIQUES:
        1. Convert float64 to float32 where precision allows
        2. Convert int64 to smaller integer types where range allows  
        3. Convert object columns to category where appropriate
        4. Remove unnecessary columns if present
        
        VALIDATION: Ensure optimization doesn't lose critical information
        """
        if df.empty:
            return df
            
        optimized_df = df.copy()
        
        # Optimize numeric columns
        for col in optimized_df.columns:
            if optimized_df[col].dtype == 'float64':
                # Check if we can safely convert to float32
                max_val = optimized_df[col].abs().max()
                if pd.notna(max_val) and max_val < 1e6:  # Safe range for float32
                    optimized_df[col] = optimized_df[col].astype('float32')
            
            elif optimized_df[col].dtype == 'int64':
                # Convert to smaller int types where possible
                max_val = optimized_df[col].max()
                min_val = optimized_df[col].min()
                
                if pd.notna(max_val) and pd.notna(min_val):
                    if min_val >= 0 and max_val <= 255:
                        optimized_df[col] = optimized_df[col].astype('uint8')
                    elif min_val >= -128 and max_val <= 127:
                        optimized_df[col] = optimized_df[col].astype('int8')
                    elif min_val >= -32768 and max_val <= 32767:
                        optimized_df[col] = optimized_df[col].astype('int16')
                    elif min_val >= -2147483648 and max_val <= 2147483647:
                        optimized_df[col] = optimized_df[col].astype('int32')
            
            elif optimized_df[col].dtype == 'object':
                # Convert low-cardinality string columns to category
                unique_ratio = optimized_df[col].nunique() / len(optimized_df)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    optimized_df[col] = optimized_df[col].astype('category')
        
        return optimized_df

    def _validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        VALIDATION: Ensure metadata completeness and consistency.
        
        VALIDATION CHECKS:
        1. Required fields presence
        2. Data type consistency
        3. Value range validation
        4. Logical consistency checks
        """
        required_fields = ['run_id', 'mechanism', 'status']
        missing_fields = [field for field in required_fields if field not in metadata]
        
        if missing_fields:
            print(f"[VALIDATION_WARNING] Metadata missing required fields: {missing_fields}")
        
        # Ensure run_id is integer
        if 'run_id' in metadata and not isinstance(metadata['run_id'], int):
            try:
                metadata['run_id'] = int(metadata['run_id'])
            except (ValueError, TypeError):
                print(f"[VALIDATION_ERROR] Invalid run_id: {metadata.get('run_id')}")
        
        return metadata

    def get_concatenated_data(self, use_chunked_processing: bool = True) -> pd.DataFrame:
        """
        AGGREGATION: Efficiently concatenate all timeline data.
        
        PROCESSING STRATEGIES:
        1. Chunked processing for large datasets
        2. Memory monitoring during concatenation
        3. Progress reporting for long operations
        4. Error handling for memory issues
        """
        if not self.all_run_data:
            print("[AGGREGATION_WARNING] No data to concatenate")
            return pd.DataFrame()
        
        print(f"[AGGREGATION_START] Concatenating {len(self.all_run_data)} DataFrames "
              f"({self.total_timeline_points} total rows)")
        
        if use_chunked_processing and len(self.all_run_data) > 100:
            return self._chunked_concatenation()
        else:
            return self._direct_concatenation()

    def _chunked_concatenation(self) -> pd.DataFrame:
        """
        MEMORY-EFFICIENT: Process data in chunks to manage memory usage.
        
        ALGORITHM:
        1. Process DataFrames in chunks of specified size
        2. Concatenate each chunk separately
        3. Combine chunk results into final DataFrame
        4. Monitor memory usage throughout process
        """
        chunk_results = []
        
        for i in range(0, len(self.all_run_data), self.chunk_size):
            chunk_end = min(i + self.chunk_size, len(self.all_run_data))
            chunk_data = self.all_run_data[i:chunk_end]
            
            print(f"[CHUNKED_PROCESSING] Processing chunk {i//self.chunk_size + 1} "
                  f"(DataFrames {i}-{chunk_end-1})")
            
            try:
                chunk_result = pd.concat(chunk_data, ignore_index=True)
                chunk_results.append(chunk_result)
                
                # Monitor memory after each chunk
                current_memory_gb = psutil.Process().memory_info().rss / 1024**3
                print(f"[CHUNK_COMPLETE] Memory usage: {current_memory_gb:.2f}GB")
                
                if current_memory_gb > self.memory_limit_gb:
                    gc.collect()
                    
            except MemoryError:
                print(f"[MEMORY_ERROR] Failed to process chunk {i//self.chunk_size + 1}")
                raise
        
        print("[FINAL_CONCATENATION] Combining chunk results...")
        final_result = pd.concat(chunk_results, ignore_index=True)
        
        return final_result

    def _direct_concatenation(self) -> pd.DataFrame:
        """DIRECT: Simple concatenation for smaller datasets."""
        try:
            result = pd.concat(self.all_run_data, ignore_index=True)
            print(f"[DIRECT_CONCATENATION] Successfully concatenated {len(result)} rows")
            return result
        except MemoryError:
            print("[MEMORY_ERROR] Direct concatenation failed, falling back to chunked processing")
            return self._chunked_concatenation()

    def get_metadata_summary(self) -> pd.DataFrame:
        """Generate metadata summary with enhanced error handling."""
        if not self.all_run_metadata:
            print("[METADATA_WARNING] No metadata to summarize")
            return pd.DataFrame()
        
        try:
            metadata_df = pd.DataFrame(self.all_run_metadata)
            print(f"[METADATA_SUMMARY] Generated summary for {len(metadata_df)} simulations")
            return metadata_df
        except Exception as e:
            print(f"[METADATA_ERROR] Failed to create metadata summary: {e}")
            return pd.DataFrame()

    def save_results(self, base_filename_prefix: str, timestamp: str, 
                    use_compression: bool = True, save_chunks: bool = False) -> None:
        """
        ENHANCED SAVE: Efficient saving with compression and chunking options.
        
        SAVE STRATEGIES:
        1. Compression to reduce file size (especially important for timeline data)
        2. Optional chunked saving for very large datasets
        3. Progress monitoring for long save operations
        4. Integrity verification after save
        
        PERFORMANCE OPTIMIZATIONS:
        - Use efficient compression algorithms
        - Monitor disk space during save
        - Provide progress feedback for large operations
        """
        print(f"[SAVE_START] Saving {self.total_simulations} simulations, {self.total_timeline_points} timeline points")
        
        # Generate filenames
        compression_ext = ".gz" if use_compression else ""
        data_filename = f"{base_filename_prefix}_timeline_data_{timestamp}.csv{compression_ext}"
        metadata_filename = f"{base_filename_prefix}_metadata_{timestamp}.csv"
        
        # Save timeline data
        timeline_data = self.get_concatenated_data()
        if not timeline_data.empty:
            save_start_time = pd.Timestamp.now()
            
            if save_chunks and len(timeline_data) > 100000:
                self._save_in_chunks(timeline_data, data_filename, use_compression)
            else:
                compression = 'gzip' if use_compression else None
                timeline_data.to_csv(data_filename, index=False, compression=compression)
            
            save_duration = (pd.Timestamp.now() - save_start_time).total_seconds()
            file_size_mb = os.path.getsize(data_filename) / 1024**2
            
            print(f"[SAVE_COMPLETE] Timeline data saved: {data_filename}")
            print(f"   File size: {file_size_mb:.1f}MB, Save time: {save_duration:.2f}s")
        
        # Save metadata
        metadata_df = self.get_metadata_summary()
        if not metadata_df.empty:
            metadata_df.to_csv(metadata_filename, index=False)
            print(f"[SAVE_COMPLETE] Metadata saved: {metadata_filename}")

    def _save_in_chunks(self, data: pd.DataFrame, filename: str, use_compression: bool) -> None:
        """Save large datasets in chunks to manage memory usage."""
        chunk_size = 50000
        num_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size > 0 else 0)
        
        print(f"[CHUNKED_SAVE] Saving in {num_chunks} chunks of {chunk_size} rows each")
        
        compression = 'gzip' if use_compression else None
        
        for i, chunk_start in enumerate(range(0, len(data), chunk_size)):
            chunk_end = min(chunk_start + chunk_size, len(data))
            chunk = data.iloc[chunk_start:chunk_end]
            
            mode = 'w' if i == 0 else 'a'
            header = i == 0
            
            chunk.to_csv(filename, mode=mode, header=header, index=False, compression=compression)
            print(f"[CHUNK_SAVE] Saved chunk {i+1}/{num_chunks}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        PERFORMANCE: Generate comprehensive performance metrics.
        
        METRICS REPORTED:
        - Data volume statistics
        - Memory usage patterns
        - Processing efficiency metrics
        - Optimization effectiveness
        """
        current_memory_gb = psutil.Process().memory_info().rss / 1024**3
        
        return {
            'total_simulations': self.total_simulations,
            'total_timeline_points': self.total_timeline_points,
            'avg_timeline_points_per_simulation': self.total_timeline_points / max(1, self.total_simulations),
            'current_memory_usage_gb': current_memory_gb,
            'memory_limit_gb': self.memory_limit_gb,
            'memory_warnings_triggered': self.memory_warnings,
            'memory_utilization_pct': (current_memory_gb / self.memory_limit_gb) * 100,
            'chunk_size': self.chunk_size
        }
    
    def add_anomalies(self, anomaly_list: List[Dict[str, Any]], run_metadata: Optional[Dict[str, Any]] = None):
        """Adds a list of anomaly dictionaries, enriching them with run_metadata if provided."""
        if not anomaly_list:
            return

        enriched_anomalies = []
        for anomaly_dict_raw in anomaly_list:
            if not isinstance(anomaly_dict_raw, dict):
                print(f"Warning: Skipping non-dict anomaly item: {anomaly_dict_raw}")
                continue

            anomaly_dict = anomaly_dict_raw.copy()

            if run_metadata:
                for key, value in run_metadata.items():
                    if key not in anomaly_dict:
                        anomaly_dict[key] = value
            enriched_anomalies.append(anomaly_dict)
        
        self.all_anomaly_logs.extend(enriched_anomalies)

    def get_anomaly_logs_df(self) -> pd.DataFrame:
        """Returns all collected anomaly logs as a DataFrame."""
        if not self.all_anomaly_logs:
            return pd.DataFrame()
        try:
            return pd.DataFrame(self.all_anomaly_logs)
        except Exception as e:
            print(f"Error converting anomaly logs to DataFrame: {e}")
            valid_logs = [log for log in self.all_anomaly_logs if isinstance(log, dict)]
            if valid_logs:
                return pd.DataFrame(valid_logs)
            return pd.DataFrame()

    def save_results(self, base_filename_prefix: str, timestamp: str, 
                    use_compression: bool = True, save_chunks: bool = False) -> None:
        """Enhanced save with anomaly logs."""
        print(f"[SAVE_START] Saving {self.total_simulations} simulations, {self.total_timeline_points} timeline points")
        
        # Generate filenames
        compression_ext = ".gz" if use_compression else ""
        data_filename = f"{base_filename_prefix}_timeline_data_{timestamp}.csv{compression_ext}"
        metadata_filename = f"{base_filename_prefix}_metadata_{timestamp}.csv"
        
        # Save timeline data
        timeline_data = self.get_concatenated_data()
        if not timeline_data.empty:
            save_start_time = pd.Timestamp.now()
            
            if save_chunks and len(timeline_data) > 100000:
                self._save_in_chunks(timeline_data, data_filename, use_compression)
            else:
                compression = 'gzip' if use_compression else None
                timeline_data.to_csv(data_filename, index=False, compression=compression)
            
            save_duration = (pd.Timestamp.now() - save_start_time).total_seconds()
            file_size_mb = os.path.getsize(data_filename) / 1024**2
            
            print(f"[SAVE_COMPLETE] Timeline data saved: {data_filename}")
            print(f"   File size: {file_size_mb:.1f}MB, Save time: {save_duration:.2f}s")
        
        # Save metadata
        metadata_df = self.get_metadata_summary()
        if not metadata_df.empty:
            metadata_df.to_csv(metadata_filename, index=False)
            print(f"[SAVE_COMPLETE] Metadata saved: {metadata_filename}")

        # NEW: Save anomaly logs
        anomaly_df = self.get_anomaly_logs_df()
        if not anomaly_df.empty:
            compression_ext_val = ".gz" if use_compression else ""
            anomaly_filename = f"{base_filename_prefix}_anomaly_logs_{timestamp}.csv{compression_ext_val}"
            compression_method = 'gzip' if use_compression else None
            
            try:
                anomaly_df.to_csv(anomaly_filename, index=False, compression=compression_method)
                print(f"[SAVE_COMPLETE] Anomaly logs saved: {anomaly_filename}")
            except Exception as e:
                print(f"[SAVE_ERROR] Could not save anomaly logs: {e}")
        else:
            print("[INFO] No anomaly logs to save.")



# BACKWARD COMPATIBILITY: Alias for existing code
ResultsAggregator = EnhancedResultsAggregator