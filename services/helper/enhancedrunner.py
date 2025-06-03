import pandas as pd
import os
import sys
from datetime import datetime
from pathlib import Path
import traceback
import warnings
import gc
from typing import Optional

# Suppress non-critical warnings to improve output clarity
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class EnhancedAnalysisRunner:
    """
    Comprehensive runner for enhanced point-specific adversarial analysis.
    
    This runner provides a robust execution framework for the enhanced analysis
    system, handling path resolution, data validation, error recovery, and 
    resource management with systematic precision.
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize runner with dynamic path resolution.
        
        Args:
            project_root: Optional explicit project root path. If None, auto-detect.
        """
        self.project_root = self._resolve_project_root(project_root)
        self._setup_import_paths()
        self.required_columns = {
            'mechanism', 'adversarial_proportion_total', 'round', 
            'resources_after', 'replication_run_index'
        }
    
    def _resolve_project_root(self, explicit_root: Optional[Path]) -> Path:
        """
        Systematically resolve project root directory with multiple fallback strategies.
        
        Args:
            explicit_root: Explicitly provided root path
            
        Returns:
            Resolved project root Path object
        """
        if explicit_root and explicit_root.exists():
            return explicit_root.resolve()
        
        try:
            # Strategy 1: Resolve from __file__ location
            script_path = Path(__file__).resolve()
            
            # Check if we're in project root
            if (script_path.parent / "services").exists():
                return script_path.parent
            
            # Check if we're one level down (experiments/, services/, etc.)
            if (script_path.parent.parent / "services").exists():
                return script_path.parent.parent
            
            # Check if we're two levels down
            if (script_path.parent.parent.parent / "services").exists():
                return script_path.parent.parent.parent
            
            # Fallback to current directory
            return script_path.parent
            
        except NameError:
            # Strategy 2: Interactive environment fallback
            current_dir = Path.cwd()
            
            # Search upward for services directory
            for parent in [current_dir] + list(current_dir.parents):
                if (parent / "services").exists():
                    return parent
            
            # Final fallback
            return current_dir
    
    def _setup_import_paths(self) -> None:
        """
        Configure Python import paths for module resolution.
        """
        project_root_str = str(self.project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
        
        # Also add services directory for direct imports
        services_path = str(self.project_root / "services")
        if services_path not in sys.path:
            sys.path.insert(0, services_path)
    
    def _validate_csv_file(self, csv_path: Path) -> bool:
        """
        Comprehensive CSV file validation with detailed error reporting.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            True if file is valid, False otherwise
        """
        if not csv_path.exists():
            print(f"❌ ERROR: CSV file does not exist: {csv_path}")
            return False
        
        if not csv_path.is_file():
            print(f"❌ ERROR: Path is not a file: {csv_path}")
            return False
        
        if csv_path.suffix.lower() != '.csv':
            print(f"⚠️  WARNING: File does not have .csv extension: {csv_path}")
        
        return True
    
    def _load_and_validate_data(self, csv_path: Path) -> Optional[pd.DataFrame]:
        """
        Load CSV data with comprehensive validation and error handling.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Loaded DataFrame if successful, None if failed
        """
        print(f"📂 Loading data from: {csv_path}")
        
        try:
            # Load data with explicit encoding handling
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            if df.empty:
                print("❌ ERROR: CSV file is empty")
                return None
            
            print(f"✅ Successfully loaded {len(df)} rows with {len(df.columns)} columns")
            
            # Validate required columns
            missing_columns = self.required_columns - set(df.columns)
            if missing_columns:
                print(f"❌ ERROR: Missing required columns: {missing_columns}")
                print(f"📋 Available columns: {list(df.columns)}")
                return None
            
            # Validate data quality
            self._validate_data_quality(df)
            
            return df
            
        except pd.errors.EmptyDataError:
            print("❌ ERROR: CSV file is empty or invalid")
            return None
        except pd.errors.ParserError as e:
            print(f"❌ ERROR: Failed to parse CSV file: {e}")
            return None
        except FileNotFoundError:
            print(f"❌ ERROR: CSV file not found: {csv_path}")
            return None
        except Exception as e:
            print(f"❌ ERROR: Unexpected error loading CSV: {e}")
            traceback.print_exc()
            return None
    
    def _validate_data_quality(self, df: pd.DataFrame) -> None:
        """
        Validate data quality and provide warnings for potential issues.
        
        Args:
            df: Loaded DataFrame
        """
        # Check for essential data characteristics
        adversarial_props = df['adversarial_proportion_total'].unique()
        mechanisms = df['mechanism'].unique()
        rounds = df['round'].unique()
        
        print(f"📊 Data Summary:")
        print(f"   • Adversarial proportions: {len(adversarial_props)} ({sorted(adversarial_props)})")
        print(f"   • Mechanisms: {len(mechanisms)} ({sorted(mechanisms)})")
        print(f"   • Rounds: {len(rounds)} (Range: {rounds.min()}-{rounds.max()})")
        
        # Check for potential data quality issues
        if len(adversarial_props) < 3:
            print("⚠️  WARNING: Very few adversarial proportions detected. Analysis may be limited.")
        
        if len(mechanisms) < 2:
            print("⚠️  WARNING: Only one mechanism detected. Comparative analysis will be limited.")
        
        # Check for missing data
        missing_resources = df['resources_after'].isna().sum()
        if missing_resources > 0:
            print(f"⚠️  WARNING: {missing_resources} missing resource values detected")
        
        # Check for negative resources
        negative_resources = (df['resources_after'] < 0).sum()
        if negative_resources > 0:
            print(f"⚠️  WARNING: {negative_resources} negative resource values detected")
    
    def _import_enhanced_analyzer(self):
        """
        Import the enhanced analysis module with comprehensive error handling.
        
        Returns:
            EnhancedHeatmapVisualizer class if successful, None if failed
        """
        try:
            # Strategy 1: Try importing from services.helper
            from services.helper.enhanced_heatmaps import EnhancedHeatmapVisualizer
            print("✅ Successfully imported enhanced analysis module from services.helper")
            return EnhancedHeatmapVisualizer
            
        except ImportError:
            try:
                # Strategy 2: Try importing from helper directly
                from helper.enhanced_heatmaps import EnhancedHeatmapVisualizer
                print("✅ Successfully imported enhanced analysis module from helper")
                return EnhancedHeatmapVisualizer
                
            except ImportError:
                try:
                    # Strategy 3: Try importing from current directory
                    from enhanced_heatmaps import EnhancedHeatmapVisualizer
                    print("✅ Successfully imported enhanced analysis module from current directory")
                    return EnhancedHeatmapVisualizer
                    
                except ImportError:
                    print("❌ ERROR: Could not import enhanced analysis module")
                    print("📋 Please ensure enhanced_heatmaps.py is available in one of these locations:")
                    print(f"   • {self.project_root}/services/helper/enhanced_heatmaps.py")
                    print(f"   • {self.project_root}/enhanced_heatmaps.py")
                    print(f"   • Current directory/enhanced_heatmaps.py")
                    return None
    
    def _create_output_directory(self, base_output_dir: str, csv_path: Path) -> Path:
        """
        Create organized output directory structure with timestamp.
        
        Args:
            base_output_dir: Base directory for outputs
            csv_path: Source CSV file path
            
        Returns:
            Created output directory path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # csv_stem = csv_path.stem # This was making the path too long
        
        # Create descriptive directory name
        analysis_dir_name = f"run_{timestamp}" # Shortened directory name
        output_dir = Path(base_output_dir) / analysis_dir_name
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created output directory: {output_dir}")
            return output_dir
            
        except Exception as e:
            print(f"❌ ERROR: Failed to create output directory: {e}")
            # Fallback to current directory with timestamp
            fallback_dir = Path.cwd() / f"enhanced_analysis_{timestamp}"
            fallback_dir.mkdir(exist_ok=True)
            print(f"📁 Using fallback directory: {fallback_dir}")
            return fallback_dir
    
    def _execute_analysis(self, df: pd.DataFrame, output_dir: Path, 
                         min_sample_size: int = 10) -> bool:
        """
        Execute enhanced analysis with comprehensive error handling and resource management.
        
        Args:
            df: Data DataFrame
            output_dir: Output directory path
            min_sample_size: Minimum sample size for statistical validity
            
        Returns:
            True if analysis completed successfully, False otherwise
        """
        # Import enhanced analyzer
        EnhancedHeatmapVisualizer = self._import_enhanced_analyzer()
        if EnhancedHeatmapVisualizer is None:
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            print("🚀 Initializing enhanced adversarial analysis...")
            
            # Create analyzer instance
            analyzer = EnhancedHeatmapVisualizer(
                timeline_df=df,
                output_dir=str(output_dir),
                min_sample_size=min_sample_size
            )
            
            print(f"📊 Analysis Configuration:")
            print(f"   • Minimum sample size: {min_sample_size}")
            print(f"   • Output directory: {output_dir}")
            print(f"   • Adversarial points detected: {len(analyzer.adversarial_points)}")
            
            # Execute complete analysis
            print("\n🎯 Executing comprehensive enhanced analysis...")
            analyzer.run_complete_enhanced_analysis(timestamp)
            
            # Memory cleanup
            del analyzer
            gc.collect()
            
            print("\n✅ Enhanced analysis completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ ERROR: Analysis execution failed: {e}")
            traceback.print_exc()
            return False
    
    def run_analysis(self, csv_file_path: str, output_directory: str, 
                    min_sample_size: int = 10) -> bool:
        """
        Execute complete enhanced adversarial analysis workflow.
        
        Args:
            csv_file_path: Path to input CSV file
            output_directory: Base directory for outputs
            min_sample_size: Minimum sample size for statistical validity
            
        Returns:
            True if analysis completed successfully, False otherwise
        """
        print("🔬 Enhanced Adversarial Analysis Runner")
        print("=" * 50)
        
        # Validate CSV file
        csv_path = Path(csv_file_path)
        if not self._validate_csv_file(csv_path):
            return False
        
        # Load and validate data
        df = self._load_and_validate_data(csv_path)
        if df is None:
            return False
        
        # Create output directory
        output_dir = self._create_output_directory(output_directory, csv_path)
        
        # Execute analysis
        success = self._execute_analysis(df, output_dir, min_sample_size)
        
        if success:
            print("\n🎉 Analysis Complete!")
            print(f"📁 Results available at: {output_dir}")
            print(f"🌐 Open {output_dir}/analysis_index.html for navigation")
        else:
            print("\n💥 Analysis failed. Check error messages above.")
        
        return success


def main_enhanced_analysis(csv_file_path: str, output_directory: str, 
                          min_sample_size: int = 10, project_root: Optional[str] = None):
    """
    Main entry point for enhanced adversarial analysis.
    
    Args:
        csv_file_path: Path to input CSV file
        output_directory: Base directory for outputs  
        min_sample_size: Minimum sample size for statistical validity
        project_root: Optional explicit project root path
    """
    try:
        # Initialize runner
        project_root_path = Path(project_root) if project_root else None
        runner = EnhancedAnalysisRunner(project_root=project_root_path)
        
        # Execute analysis
        success = runner.run_analysis(
            csv_file_path=csv_file_path,
            output_directory=output_directory,
            min_sample_size=min_sample_size
        )
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # ========================================
    # USER CONFIGURATION SECTION
    # ========================================
    
    # INPUT: Path to your aggregated timeline CSV file
    CSV_TO_ANALYZE = r"experiment_outputs\TimelinePortfolioDemocracySuite_20250603_161822\StableBaseline_DevilsAdvocate_20250603_161822\DevilsAdvocate.csv"
    
    # OUTPUT: Base directory where analysis results will be saved
    # A timestamped subdirectory will be created automatically
    BASE_OUTPUT_DIR = r"experiment_outputs\TimelinePortfolioDemocracySuite_20250603_161822\enhanced_analysis_competitive_DevilsAdvocate"
    
    # CONFIGURATION: Minimum sample size for statistical validity
    # Adversarial proportions with fewer samples will be flagged with warnings
    MIN_SAMPLE_SIZE = 5
    
    # OPTIONAL: Explicit project root path (leave as None for auto-detection)
    PROJECT_ROOT = None  # e.g., r"C:\Users\Jonas\Documents\GitHub\Mycorrhiza"
    
    # ========================================
    # END USER CONFIGURATION
    # ========================================
    
    # Validate configuration
    if not os.path.exists(CSV_TO_ANALYZE):
        print("❌ ERROR: The specified CSV file does not exist!")
        print(f"📂 Specified path: {CSV_TO_ANALYZE}")
        print("📝 Please update the 'CSV_TO_ANALYZE' variable in the configuration section.")
        print("\n💡 Tip: Use forward slashes (/) or raw strings (r\"\") for Windows paths")
        sys.exit(1)
    
    # Execute analysis
    print("🚀 Starting Enhanced Adversarial Analysis...")
    print(f"📂 Input CSV: {CSV_TO_ANALYZE}")
    print(f"📁 Output Directory: {BASE_OUTPUT_DIR}")
    print(f"📊 Minimum Sample Size: {MIN_SAMPLE_SIZE}")
    
    main_enhanced_analysis(
        csv_file_path=CSV_TO_ANALYZE,
        output_directory=BASE_OUTPUT_DIR,
        min_sample_size=MIN_SAMPLE_SIZE,
        project_root=PROJECT_ROOT
    )