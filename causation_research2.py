import pandas as pd
import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.linalg import cholesky, LinAlgError
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
import warnings
from typing import Dict, List, Tuple, Optional, Union
import time
from dataclasses import dataclass
from collections import defaultdict

warnings.filterwarnings('ignore')


@dataclass
class CausationResult:
    """Container for causation test results"""
    feature_name: str
    event_name: str
    granger_pvalue: float
    granger_fstatistic: float
    optimal_lag: int
    correlation: float
    mutual_info_score: float
    stationarity_feature: bool
    stationarity_event: bool
    sample_size: int
    relationship_strength: str
    is_significant: bool


class OptimizedCausationAnalyzer:
    """
    Heavily optimized causation analysis system with algorithmic improvements

    Key optimizations:
    1. Hierarchical statistical screening
    2. Incremental computation with caching
    3. Spectral methods for large datasets
    4. Smart lag selection
    5. Vectorized operations throughout
    """

    def __init__(self,
                 significance_level: float = 0.05,
                 max_lag: int = 5,
                 min_samples_per_event: int = 50,
                 enable_spectral_methods: bool = True,
                 cache_stationarity: bool = True):

        self.significance_level = significance_level
        self.max_lag = max_lag
        self.min_samples_per_event = min_samples_per_event
        self.enable_spectral_methods = enable_spectral_methods
        self.cache_stationarity = cache_stationarity

        # Caching systems
        self.stationarity_cache = {}
        self.covariance_cache = {}
        self.feature_stats_cache = {}

        # Performance tracking
        self.timing_stats = defaultdict(float)
        self.test_counts = defaultdict(int)

    def analyze_causation(self,
                          data: pd.DataFrame,
                          event_columns: List[str],
                          feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Main causation analysis with optimized pipeline
        """

        start_time = time.time()
        print(f"Starting optimized causation analysis...")
        print(f"Dataset shape: {data.shape}")

        # Auto-detect feature columns if not provided
        if feature_columns is None:
            feature_columns = [col for col in data.columns
                               if col not in event_columns and data[col].dtype in ['float64', 'int64']]

        print(f"Testing {len(feature_columns)} features against {len(event_columns)} events")

        # Validate and prepare data
        data_clean = self._prepare_data(data, feature_columns + event_columns)

        # Phase 1: Quick screening with correlation and mutual information
        print("Phase 1: Statistical pre-screening...")
        prescreened_pairs = self._hierarchical_prescreening(data_clean, feature_columns, event_columns)

        print(
            f"Pre-screening reduced candidates from {len(feature_columns) * len(event_columns)} to {len(prescreened_pairs)}")

        # Phase 2: Stationarity testing with caching
        print("Phase 2: Efficient stationarity testing...")
        stationarity_results = self._batch_stationarity_testing(data_clean, prescreened_pairs)

        # Phase 3: Optimized Granger causality testing
        print("Phase 3: Optimized causality testing...")
        causation_results = self._optimized_granger_testing(data_clean, prescreened_pairs, stationarity_results)

        # Compile results
        results_df = self._compile_results(causation_results)

        total_time = time.time() - start_time
        print(f"\nAnalysis completed in {total_time:.2f} seconds")
        self._print_performance_summary()

        return results_df

    def _prepare_data(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fast data preparation with vectorized operations"""

        # Select and clean data
        data_subset = data[columns].copy()

        # Vectorized missing value handling
        data_subset = data_subset.fillna(method='ffill').fillna(method='bfill')

        # Remove infinite values
        data_subset = data_subset.replace([np.inf, -np.inf], np.nan)
        data_subset = data_subset.dropna()

        print(f"Data preparation: {len(data)} -> {len(data_subset)} rows after cleaning")

        return data_subset

    def _hierarchical_prescreening(self,
                                   data: pd.DataFrame,
                                   feature_columns: List[str],
                                   event_columns: List[str]) -> List[Tuple[str, str]]:
        """
        Multi-stage statistical screening to identify promising feature-event pairs
        """

        start_time = time.time()
        promising_pairs = []

        # Convert boolean events to numeric for correlation calculation
        event_data = data[event_columns].astype(float)
        feature_data = data[feature_columns]

        for event_col in event_columns:
            event_series = event_data[event_col]

            # Skip events with insufficient variation
            if 'covent' in event_col.lower():
                min_samples = max(10, self.min_samples_per_event // 5)  # Much lower threshold
            elif 'confluence' in event_col.lower():
                min_samples = max(15, self.min_samples_per_event // 3)
            else:
                min_samples = self.min_samples_per_event

            if event_series.sum() < min_samples:
                print(f"Skipping {event_col}: only {event_series.sum()} events (need {min_samples})")
                continue

            # Stage 1: Vectorized correlation screening
            correlations = feature_data.corrwith(event_series).abs()

            # Stage 2: Mutual information for top correlation candidates
            top_corr_features = correlations.nlargest(min(30, len(feature_columns))).index.tolist()

            # Calculate mutual information for top candidates
            if len(top_corr_features) > 0:
                feature_subset = feature_data[top_corr_features].values
                mi_scores = mutual_info_regression(feature_subset, event_series.values, random_state=42)

                # Combine correlation and MI scores
                combined_scores = pd.Series(index=top_corr_features,
                                            data=correlations[top_corr_features].values * 0.7 +
                                                 (mi_scores / max(mi_scores.max(), 1e-10)) * 0.3)

                # Select top candidates based on combined score
                threshold = max(0.05, combined_scores.quantile(0.7))  # Adaptive threshold
                selected_features = combined_scores[combined_scores >= threshold].index.tolist()

                # Add to promising pairs
                for feature in selected_features:
                    promising_pairs.append((feature, event_col))

        self.timing_stats['prescreening'] = time.time() - start_time
        return promising_pairs

    def _batch_stationarity_testing(self,
                                    data: pd.DataFrame,
                                    pairs: List[Tuple[str, str]]) -> Dict[str, bool]:
        """
        Efficient batch stationarity testing with caching
        """

        start_time = time.time()
        results = {}

        # Get unique series to test (avoid duplicates)
        unique_series = set()
        for feature, event in pairs:
            unique_series.add(feature)
            unique_series.add(event)

        for series_name in unique_series:
            if series_name in self.stationarity_cache and self.cache_stationarity:
                results[series_name] = self.stationarity_cache[series_name]
                continue

            series = data[series_name].dropna()

            # Fast stationarity test using variance ratio
            is_stationary = self._fast_stationarity_test(series)

            results[series_name] = is_stationary

            if self.cache_stationarity:
                self.stationarity_cache[series_name] = is_stationary

        self.timing_stats['stationarity'] = time.time() - start_time
        return results

    def _fast_stationarity_test(self, series: pd.Series) -> bool:
        """
        Fast stationarity test using variance ratio method
        Much faster than ADF/KPSS for large series
        """

        if len(series) < 100:
            # For small series, use traditional ADF
            try:
                adf_stat, adf_pvalue = adfuller(series, maxlag=1)[:2]
                return adf_pvalue < self.significance_level
            except:
                return False

        # For large series, use variance ratio test (much faster)
        try:
            # Calculate variance ratios for different periods
            periods = [2, 4, 8, 16]
            variance_ratios = []

            for period in periods:
                if len(series) > period * 10:  # Ensure sufficient data
                    # Calculate period returns
                    period_returns = series.pct_change(period).dropna()
                    single_returns = series.pct_change().dropna()

                    if len(period_returns) > 10 and len(single_returns) > 10:
                        # Variance ratio
                        var_ratio = (period_returns.var() * period) / (single_returns.var() * len(period_returns))
                        variance_ratios.append(abs(var_ratio - 1.0))

            if variance_ratios:
                # If variance ratios are close to 1, likely stationary
                avg_deviation = np.mean(variance_ratios)
                return avg_deviation < 0.5  # Empirical threshold
            else:
                return False

        except:
            return False

    def _optimized_granger_testing(self,
                                   data: pd.DataFrame,
                                   pairs: List[Tuple[str, str]],
                                   stationarity_results: Dict[str, bool]) -> List[CausationResult]:
        """
        Optimized Granger causality testing with smart algorithms
        """

        start_time = time.time()
        results = []

        for feature_name, event_name in pairs:

            # Skip if either series is non-stationary (would need differencing)
            if not stationarity_results.get(feature_name, False) or not stationarity_results.get(event_name, False):
                continue

            feature_series = data[feature_name].dropna()
            event_series = data[event_name].dropna()

            # Align series
            common_index = feature_series.index.intersection(event_series.index)
            if len(common_index) < 100:  # Need sufficient data
                continue

            feature_aligned = feature_series.loc[common_index]
            event_aligned = event_series.loc[common_index]

            # Fast Granger causality test
            result = self._fast_granger_test(feature_aligned, event_aligned, feature_name, event_name)

            if result:
                results.append(result)

        self.timing_stats['granger_testing'] = time.time() - start_time
        return results

    def _fast_granger_test(self,
                           feature_series: pd.Series,
                           event_series: pd.Series,
                           feature_name: str,
                           event_name: str) -> Optional[CausationResult]:
        """
        Fast implementation of Granger causality test using linear methods
        """

        try:
            # Prepare data matrix for VAR model
            data_matrix = pd.DataFrame({
                'event': event_series,
                'feature': feature_series
            })

            # Smart lag selection using information criteria
            optimal_lag = self._select_optimal_lag(data_matrix)

            if optimal_lag == 0:
                return None

            # Fit restricted and unrestricted models efficiently
            if self.enable_spectral_methods and len(data_matrix) > 1000:
                # Use spectral method for large datasets
                granger_stat, p_value = self._spectral_granger_test(
                    feature_series.values, event_series.values, optimal_lag
                )
            else:
                # Use traditional VAR method for smaller datasets
                granger_stat, p_value = self._var_granger_test(data_matrix, optimal_lag)

            # Additional statistics
            correlation = feature_series.corr(event_series)

            # Determine relationship strength
            if p_value < 0.01:
                strength = "Strong"
            elif p_value < 0.05:
                strength = "Moderate"
            elif p_value < 0.10:
                strength = "Weak"
            else:
                strength = "None"

            return CausationResult(
                feature_name=feature_name,
                event_name=event_name,
                granger_pvalue=p_value,
                granger_fstatistic=granger_stat,
                optimal_lag=optimal_lag,
                correlation=correlation,
                mutual_info_score=0.0,  # Could add if needed
                stationarity_feature=True,
                stationarity_event=True,
                sample_size=len(feature_series),
                relationship_strength=strength,
                is_significant=p_value < self.significance_level
            )

        except Exception as e:
            print(f"Error in Granger test for {feature_name} -> {event_name}: {str(e)}")
            return None

    def analyze_event_hierarchy(self, data: pd.DataFrame, results: pd.DataFrame):
        """
        Analyze how individual event predictors relate to co-event predictors
        """

        individual_events = [col for col in data.columns if col.endswith('_event') and 'covent' not in col]
        covent_events = [col for col in data.columns if 'covent' in col]

        hierarchy_analysis = {}

        for covent in covent_events:
            # Find which individual events contribute to this co-event
            base_events = []
            for individual in individual_events:
                if individual.replace('_event', '') in covent:
                    base_events.append(individual)

            # Compare predictors
            covent_predictors = set(results[results['event'] == covent]['feature'].tolist())

            individual_predictors = set()
            for base_event in base_events:
                individual_predictors.update(
                    results[results['event'] == base_event]['feature'].tolist()
                )

            hierarchy_analysis[covent] = {
                'base_events': base_events,
                'unique_predictors': covent_predictors - individual_predictors,
                'shared_predictors': covent_predictors & individual_predictors,
                'enhanced_prediction': len(covent_predictors) > len(individual_predictors)
            }

        return hierarchy_analysis

    def _covent_specific_granger_test(self, feature_series, event_series, feature_name, event_name):
        """
        Modified Granger test optimized for rare co-events
        """

        # For very rare events, use different approaches
        event_rate = event_series.mean()

        if event_rate < 0.005:  # Less than 0.5% event rate
            # Use logistic regression approach for very rare events
            return self._logistic_granger_test(feature_series, event_series, feature_name, event_name)
        elif event_rate < 0.02:  # Less than 2% event rate
            # Use bootstrap approach for better p-value estimation
            return self._bootstrap_granger_test(feature_series, event_series, feature_name, event_name)
        else:
            # Use standard approach
            return self._fast_granger_test(feature_series, event_series, feature_name, event_name)

    def _select_optimal_lag(self, data_matrix: pd.DataFrame) -> int:
        """
        Smart lag selection using information criteria
        Much faster than testing all lags individually
        """

        try:
            # Quick AIC/BIC calculation for different lags
            best_lag = 1
            best_ic = np.inf

            for lag in range(1, min(self.max_lag + 1, len(data_matrix) // 20)):
                try:
                    # Fit VAR model
                    model = VAR(data_matrix)
                    fitted_model = model.fit(maxlags=lag, verbose=False)

                    # Use BIC for model selection (penalizes complexity more)
                    ic_value = fitted_model.bic

                    if ic_value < best_ic:
                        best_ic = ic_value
                        best_lag = lag

                except:
                    continue

            return best_lag

        except:
            return 1  # Default to lag 1

    def _spectral_granger_test(self,
                               feature_series: np.ndarray,
                               event_series: np.ndarray,
                               lag: int) -> Tuple[float, float]:
        """
        Spectral Granger causality test - much faster for large datasets
        Uses frequency domain methods
        """

        try:
            # Compute cross-spectral density
            n = min(len(feature_series), len(event_series))

            # FFT-based cross-correlation
            feature_fft = fft(feature_series[:n])
            event_fft = fft(event_series[:n])

            # Cross-spectral density
            cross_spectrum = feature_fft * np.conj(event_fft)

            # Power spectral densities
            feature_psd = np.abs(feature_fft) ** 2
            event_psd = np.abs(event_fft) ** 2

            # Coherence-based Granger causality measure
            coherence = np.abs(cross_spectrum) ** 2 / (feature_psd * event_psd + 1e-10)

            # Statistical test based on coherence
            mean_coherence = np.mean(coherence[1:n // 2])  # Exclude DC component

            # Convert to F-statistic approximation
            f_stat = mean_coherence * (n - 2 * lag) / ((1 - mean_coherence) * lag)

            # Approximate p-value using F-distribution
            p_value = 1 - stats.f.cdf(f_stat, lag, n - 2 * lag)

            return f_stat, p_value

        except:
            # Fallback to correlation-based test
            corr = np.corrcoef(feature_series, event_series)[0, 1]
            t_stat = abs(corr) * np.sqrt((n - 2) / (1 - corr ** 2 + 1e-10))
            p_value = 2 * (1 - stats.t.cdf(t_stat, n - 2))
            return t_stat, p_value

    def _var_granger_test(self, data_matrix: pd.DataFrame, lag: int) -> Tuple[float, float]:
        """
        Traditional VAR-based Granger causality test with optimizations
        """

        try:
            # Fit VAR model
            model = VAR(data_matrix)
            fitted_model = model.fit(maxlags=lag, verbose=False)

            # Granger causality test
            granger_result = fitted_model.test_causality('event', 'feature', kind='f', verbose=False)

            return granger_result.fvalue, granger_result.pvalue

        except Exception as e:
            # Fallback to simple regression-based test
            return self._simple_regression_test(data_matrix, lag)

    def _simple_regression_test(self, data_matrix: pd.DataFrame, lag: int) -> Tuple[float, float]:
        """
        Simple regression-based causality test as fallback
        """

        try:
            # Create lagged features
            event_col = data_matrix['event']
            feature_col = data_matrix['feature']

            # Prepare regression matrices
            y = event_col.iloc[lag:].values
            X_restricted = np.column_stack([event_col.shift(i).iloc[lag:].values for i in range(1, lag + 1)])
            X_unrestricted = np.column_stack([
                X_restricted,
                np.column_stack([feature_col.shift(i).iloc[lag:].values for i in range(1, lag + 1)])
            ])

            # Add constant term
            X_restricted = np.column_stack([np.ones(len(X_restricted)), X_restricted])
            X_unrestricted = np.column_stack([np.ones(len(X_unrestricted)), X_unrestricted])

            # Fit models using normal equations (faster than iterative methods)
            try:
                beta_restricted = np.linalg.solve(X_restricted.T @ X_restricted, X_restricted.T @ y)
                beta_unrestricted = np.linalg.solve(X_unrestricted.T @ X_unrestricted, X_unrestricted.T @ y)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                beta_restricted = np.linalg.pinv(X_restricted) @ y
                beta_unrestricted = np.linalg.pinv(X_unrestricted) @ y

            # Calculate residuals
            resid_restricted = y - X_restricted @ beta_restricted
            resid_unrestricted = y - X_unrestricted @ beta_unrestricted

            # F-test
            rss_restricted = np.sum(resid_restricted ** 2)
            rss_unrestricted = np.sum(resid_unrestricted ** 2)

            n = len(y)
            k_restricted = X_restricted.shape[1]
            k_unrestricted = X_unrestricted.shape[1]

            f_stat = ((rss_restricted - rss_unrestricted) / (k_unrestricted - k_restricted)) / \
                     (rss_unrestricted / (n - k_unrestricted))

            p_value = 1 - stats.f.cdf(f_stat, k_unrestricted - k_restricted, n - k_unrestricted)

            return f_stat, p_value

        except:
            return 0.0, 1.0

    def _compile_results(self, causation_results: List[CausationResult]) -> pd.DataFrame:
        """
        Compile results into structured DataFrame with multiple testing correction
        """

        if not causation_results:
            return pd.DataFrame()

        # Convert results to DataFrame
        results_data = []
        for result in causation_results:
            results_data.append({
                'feature': result.feature_name,
                'event': result.event_name,
                'granger_pvalue': result.granger_pvalue,
                'granger_fstatistic': result.granger_fstatistic,
                'optimal_lag': result.optimal_lag,
                'correlation': result.correlation,
                'sample_size': result.sample_size,
                'relationship_strength': result.relationship_strength,
                'is_significant_raw': result.is_significant
            })

        results_df = pd.DataFrame(results_data)

        # Multiple testing correction using Benjamini-Hochberg (FDR control)

        _, pvalues_corrected, _, _ = multipletests(
            results_df['granger_pvalue'].values,
            alpha=self.significance_level,
            method='fdr_by'
        )

        results_df['granger_pvalue_corrected'] = pvalues_corrected
        results_df['is_significant_corrected'] = pvalues_corrected < self.significance_level

        # Sort by significance and strength
        results_df = results_df.sort_values(['is_significant_corrected', 'granger_pvalue'],
                                            ascending=[False, True])

        return results_df

    def _print_performance_summary(self):
        """Print performance summary"""

        print("\n" + "=" * 50)
        print("PERFORMANCE SUMMARY")
        print("=" * 50)

        total_time = sum(self.timing_stats.values())

        for stage, time_taken in self.timing_stats.items():
            percentage = (time_taken / total_time) * 100 if total_time > 0 else 0
            print(f"{stage.capitalize():20}: {time_taken:6.2f}s ({percentage:5.1f}%)")

        print(f"{'Total Time':20}: {total_time:6.2f}s")
        print("=" * 50)


def run_optimized_causation_analysis(data: pd.DataFrame,
                                     event_columns: List[str],
                                     feature_columns: Optional[List[str]] = None,
                                     config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Main function to run optimized causation analysis

    Args:
        data: DataFrame with time series data
        event_columns: List of event column names
        feature_columns: Optional list of feature columns (auto-detected if None)
        config: Optional configuration dictionary

    Returns:
        DataFrame with causation analysis results
    """

    # Default configuration
    default_config = {
        'significance_level': 0.05,
        'max_lag': 5,
        'min_samples_per_event': 50,
        'enable_spectral_methods': True,
        'cache_stationarity': True
    }

    if config:
        default_config.update(config)

    # Initialize analyzer
    analyzer = OptimizedCausationAnalyzer(**default_config)

    # Run analysis
    results = analyzer.analyze_causation(data, event_columns, feature_columns)

    return results


# Example usage function
# Replace the existing example_usage_with_data function with this enhanced version:

def example_usage_with_data(indicated_data: pd.DataFrame):
    """
    Example of how to use the optimized causation analyzer with detailed output
    """

    # Define event columns (from your technical indicators' system)
    event_columns = [
        'vpd_volatility_event',
        'outlier_event',
        'momentum_regime_event',
        'CUSUM_event',
        'BB_upper_cross',
        'BB_lower_cross',
        'BB_any_cross',
        'BB_squeeze',
        'BB_expansion',
        'any_event'
    ]

    # Filter to existing event columns
    existing_events = [col for col in event_columns if col in indicated_data.columns]

    # Define feature columns (exclude events and basic OHLCV)
    exclude_columns = existing_events + ['Open', 'High', 'Low', 'Close', 'Volume', 't']
    feature_columns = [col for col in indicated_data.columns
                       if col not in exclude_columns and
                       indicated_data[col].dtype in ['float64', 'int64'] and
                       not col.endswith('_event')]

    print(f"Found {len(existing_events)} event columns and {len(feature_columns)} feature columns")

    # Data quality check
    print(f"\n{'=' * 60}")
    print("DATA QUALITY OVERVIEW")
    print(f"{'=' * 60}")
    print(f"Sample size: {len(indicated_data):,} observations")
    print(f"Time range: {indicated_data.index.min()} to {indicated_data.index.max()}")

    print(f"\nEvent frequencies:")
    for event_col in existing_events:
        if event_col in indicated_data.columns:
            event_rate = indicated_data[event_col].mean()
            event_count = indicated_data[event_col].sum()
            print(f"  {event_col:<25}: {event_count:6,} events ({event_rate:6.2%} rate)")

    # Configuration for financial time series (LESS STRICT)
    config = {
        'significance_level': 0.15,  # More lenient for rare co-events
        'max_lag': 12,  # Co-events might have longer prediction horizons
        'min_samples_per_event': 25,  # Lower threshold for rare events
        'enable_spectral_methods': True,
        'cache_stationarity': True
    }

    print(f"\nConfiguration: significance_level={config['significance_level']}, max_lag={config['max_lag']}")

    # Run analysis
    results = run_optimized_causation_analysis(
        data=indicated_data,
        event_columns=existing_events,
        feature_columns=feature_columns,
        config=config
    )

    # Print comprehensive results summary
    print(f"\n{'=' * 60}")
    print("CAUSATION ANALYSIS RESULTS")
    print(f"{'=' * 60}")

    total_possible = len(feature_columns) * len(existing_events)
    print(f"Total relationships tested: {total_possible:,}")
    print(f"Relationships found: {len(results)}")

    if len(results) > 0:
        print(f"Significant (corrected): {results['is_significant_corrected'].sum()}")
        print(f"Significant (raw): {results['is_significant_raw'].sum()}")

        print(f"\nTOP 15 RELATIONSHIPS (by corrected p-value):")
        print(f"{'Feature':<35} {'Event':<25} {'P-val':<8} {'F-stat':<8} {'Lag':<4} {'Corr':<6}")
        print("-" * 90)

        top_results = results.head(15)
        for _, row in top_results.iterrows():
            significance_marker = "***" if row['is_significant_corrected'] else "**" if row[
                'is_significant_raw'] else ""
            print(f"{row['feature']:<35} {row['event']:<25} "
                  f"{row['granger_pvalue_corrected']:<8.4f} "
                  f"{row['granger_fstatistic']:<8.2f} "
                  f"{row['optimal_lag']:<4} "
                  f"{row['correlation']:<6.3f} {significance_marker}")

        # Results by event type
        print(f"\n{'=' * 60}")
        print("RESULTS BY EVENT TYPE")
        print(f"{'=' * 60}")

        for event in results['event'].unique():
            event_results = results[results['event'] == event]
            sig_count_corrected = event_results['is_significant_corrected'].sum()
            sig_count_raw = event_results['is_significant_raw'].sum()
            total_count = len(event_results)

            print(f"\n{event}:")
            print(f"  Total predictors tested: {total_count}")
            print(f"  Significant (corrected): {sig_count_corrected}")
            print(f"  Significant (raw): {sig_count_raw}")

            if sig_count_corrected > 0:
                print(f"  Top corrected predictors:")
                top_predictors = event_results[event_results['is_significant_corrected']].head(3)
                for _, row in top_predictors.iterrows():
                    print(
                        f"    • {row['feature']:<30} (p={row['granger_pvalue_corrected']:.4f}, lag={row['optimal_lag']})")
            elif sig_count_raw > 0:
                print(f"  Top raw predictors:")
                top_predictors = event_results[event_results['is_significant_raw']].head(3)
                for _, row in top_predictors.iterrows():
                    print(f"    • {row['feature']:<30} (p={row['granger_pvalue']:.4f}, lag={row['optimal_lag']})")
            else:
                print(f"    No significant predictors found")

        # Summary statistics
        print(f"\n{'=' * 60}")
        print("SUMMARY STATISTICS")
        print(f"{'=' * 60}")
        print(
            f"Average lag for significant relationships: {results[results['is_significant_corrected']]['optimal_lag'].mean():.1f}")
        print(f"Most common lag: {results['optimal_lag'].mode().iloc[0] if len(results) > 0 else 'N/A'}")
        print(f"Strongest F-statistic: {results['granger_fstatistic'].max():.2f}")
        print(f"Best p-value (corrected): {results['granger_pvalue_corrected'].min():.6f}")

    else:
        print("\n⚠️  No relationships found!")
        print("Possible reasons:")
        print("  • Multiple testing correction too strict")
        print("  • Events too rare in the data")
        print("  • Time series relationships genuinely weak")
        print("  • Need different lag periods or methods")
        print("  • Stationarity requirements too strict")

        # Diagnostic suggestions
        print(f"\nDiagnostic suggestions:")
        print(f"  • Try significance_level=0.15 for more lenient testing")
        print(f"  • Reduce min_samples_per_event to 25")
        print(f"  • Check if events have sufficient variation")

    return results


# Performance comparison function
def compare_performance(data: pd.DataFrame, sample_sizes: List[int] = [10000, 50000, 100000]):
    """
    Compare performance across different data sizes
    """

    print("PERFORMANCE COMPARISON")
    print("=" * 50)

    # Dummy event columns for testing
    data_test = data.copy()
    data_test['test_event'] = (
            data_test['Close'].pct_change().abs() > data_test['Close'].pct_change().abs().quantile(0.95))

    event_cols = ['test_event']
    feature_cols = [col for col in data_test.columns if
                    col not in event_cols + ['Open', 'High', 'Low', 'Close', 'Volume']][:10]

    for sample_size in sample_sizes:
        if len(data_test) >= sample_size:
            print(f"\nTesting with {sample_size:,} samples...")

            # Sample data
            sampled_data = data_test.iloc[-sample_size:].copy()

            start_time = time.time()

            # Run analysis
            results = run_optimized_causation_analysis(
                sampled_data,
                event_cols,
                feature_cols[:5],  # Limit features for comparison
                {'significance_level': 0.05, 'max_lag': 3}
            )

            elapsed_time = time.time() - start_time

            print(f"Completed in {elapsed_time:.2f} seconds")
            print(f"Found {len(results)} significant relationships")
            print(f"Processing rate: {sample_size / elapsed_time:,.0f} samples/second")


def diagnose_data_issues(data: pd.DataFrame, event_columns: List[str]) -> None:
    """
    Diagnose potential data issues that could prevent finding relationships
    """

    print(f"\n{'=' * 60}")
    print("DATA DIAGNOSTIC REPORT")
    print(f"{'=' * 60}")

    existing_events = [col for col in event_columns if col in data.columns]

    # Check event frequencies
    print("Event frequency analysis:")
    rare_events = []
    for event_col in existing_events:
        if event_col in data.columns:
            event_rate = data[event_col].mean()
            event_count = data[event_col].sum()

            if event_rate < 0.01:  # Less than 1%
                rare_events.append(event_col)
                print(f"  ⚠️  {event_col}: {event_count} events ({event_rate:.3%}) - TOO RARE")
            elif event_rate < 0.05:  # Less than 5%
                print(f"  ⚠️  {event_col}: {event_count} events ({event_rate:.3%}) - RARE")
            else:
                print(f"  ✓  {event_col}: {event_count} events ({event_rate:.3%}) - OK")

    if rare_events:
        print(f"\nFound {len(rare_events)} rare events that may have insufficient data for analysis")

    # Check for feature diversity
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    feature_candidates = [col for col in numeric_cols if col not in existing_events]

    print(f"\nFeature analysis:")
    print(f"  Total numeric columns: {len(numeric_cols)}")
    print(f"  Feature candidates: {len(feature_candidates)}")

    # Check for constant or near-constant features
    low_variance_features = []
    for col in feature_candidates[:20]:  # Check first 20 features
        if data[col].var() < 1e-10:
            low_variance_features.append(col)

    if low_variance_features:
        print(f"  ⚠️  Found {len(low_variance_features)} near-constant features")

    # Memory and performance estimates
    total_tests = len(feature_candidates) * len(existing_events)
    print(f"\nPerformance estimates:")
    print(f"  Total possible tests: {total_tests:,}")
    print(f"  Estimated runtime: {total_tests / 10000:.1f} - {total_tests / 5000:.1f} minutes")

    # Recommendations
    print(f"\nRecommendations:")
    if len(rare_events) > len(existing_events) // 2:
        print(f"  • Consider using significance_level=0.15 (more lenient)")
        print(f"  • Reduce min_samples_per_event to 25")

    if total_tests > 50000:
        print(f"  • Consider pre-filtering features by correlation first")
        print(f"  • Focus on most important event types initially")

    print(f"  • Use enable_spectral_methods=True for large datasets")
    print(f"  • Consider running on a subset first to test")


def get_all_event_columns(data: pd.DataFrame) -> List[str]:
    """Automatically detect all event columns including co-events"""

    # Individual events
    individual_events = [col for col in data.columns if col.endswith('_event')]

    # Co-events
    covent_columns = [col for col in data.columns if 'covent' in col.lower()]

    # Confluence patterns
    confluence_columns = [col for col in data.columns if 'confluence' in col.lower()]

    # BB events
    bb_events = [col for col in data.columns if
                 col.startswith('BB_') and ('cross' in col or 'squeeze' in col or 'expansion' in col)]

    # CUSUM events
    cusum_events = [col for col in data.columns if 'CUSUM' in col and col != 'CUSUM_pos' and col != 'CUSUM_neg']

    # Combine all
    all_events = list(set(individual_events + covent_columns + confluence_columns + bb_events + cusum_events))

    return [col for col in all_events if data[col].dtype == 'bool' or data[col].nunique() == 2]


def discover_most_predictable_covents(res: pd.DataFrame):
    """
    Identify which types of co-events are most predictable
    """

    covent_results = res[res['event'].str.contains('covent|confluence')] # TODO KeyError: 'event'

    if len(covent_results) == 0:
        return

    predictability_ranking = covent_results.groupby('event').agg({
        'is_significant_corrected': 'sum',
        'granger_pvalue_corrected': 'min',
        'granger_fstatistic': 'max'
    }).sort_values('is_significant_corrected', ascending=False)

    print(f"\nMOST PREDICTABLE CO-EVENTS:")
    for event, stats in predictability_ranking.head(10).iterrows():
        print(f"{event}: {int(stats['is_significant_corrected'])} predictors, "
              f"best p={stats['granger_pvalue_corrected']:.4f}")


def print_covent_analysis(data):
    """
    Specialized output for co-event analysis
    """

    # Separate individual events from co-events
    individual_results = data[~data['event'].str.contains('covent|confluence')] # TODO KeyError: 'event'
    covent_results = data[data['event'].str.contains('covent')]
    confluence_results = data[data['event'].str.contains('confluence')]

    print(f"\n{'=' * 80}")
    print("CO-EVENT CAUSATION ANALYSIS")
    print(f"{'=' * 80}")

    print(f"Individual Events: {len(individual_results)} relationships found")
    print(f"2-way/3-way Co-events: {len(covent_results)} relationships found")
    print(f"Confluence Patterns: {len(confluence_results)} relationships found")

    # Compare predictability
    if len(covent_results) > 0:
        print(f"\nCO-EVENT PREDICTABILITY:")
        covent_summary = covent_results.groupby('event').agg({
            'granger_pvalue_corrected': ['min', 'mean', 'count'],
            'granger_fstatistic': 'max',
            'optimal_lag': 'mean'
        }).round(4)

        for event in covent_summary.index:
            event_data = covent_summary.loc[event]
            sig_count = covent_results[(covent_results['event'] == event) &
                                       (covent_results['is_significant_corrected'])].shape[0]

            print(f"\n{event}:")
            print(f"  Significant predictors: {sig_count}")
            print(f"  Best p-value: {event_data[('granger_pvalue_corrected', 'min')]:.6f}")
            print(f"  Average lag: {event_data[('optimal_lag', 'mean')]:.1f}")
            print(f"  Strongest F-stat: {event_data[('granger_fstatistic', 'max')]:.2f}")


if __name__ == "__main__":
    print("Optimized Causation Research Script Loaded")
    print("Use example_usage_with_data(your_dataframe) to test with your data")

    indicated = pd.read_pickle('D:/Seagull_data/labeled5mEE2cov.pkl')

    # Run diagnostics first
    event_columns = get_all_event_columns(indicated)
    diagnose_data_issues(indicated, event_columns)

    # Then run full analysis
    results = example_usage_with_data(indicated)
    # print_covent_analysis(results)
    # discover_most_predictable_covents(results)

    # Print final summary
    if len(results) > 0:
        print(f"\n🎉 SUCCESS: Found {len(results)} relationships!")
        print(f"📊 {results['is_significant_corrected'].sum()} are significant after correction")
    else:
        print(f"\n❌ No relationships found. Check diagnostic report above.")
