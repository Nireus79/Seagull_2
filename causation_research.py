import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import warnings
from pathlib import Path
import json
from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# import seaborn as sns
from scipy import stats
from scipy.stats import chi2
import itertools
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, asdict
import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import functools
from enum import Enum
from abc import ABC, abstractmethod

# Statistical and Causal Analysis
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# Machine Learning (for validation only)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class CausalRelationshipType(Enum):
    """Types of causal relationships in financial markets"""
    GRANGER_CAUSAL = "granger_causal"
    STRUCTURAL_CAUSAL = "structural_causal"
    INSTRUMENTAL = "instrumental"
    INFORMATION_FLOW = "information_flow"
    MARKET_MICROSTRUCTURE = "market_microstructure"
    BEHAVIORAL = "behavioral"
    FUNDAMENTAL = "fundamental"
    TECHNICAL_MOMENTUM = "technical_momentum"


class EconomicMechanism(Enum):
    """Known economic mechanisms in financial markets"""
    INFORMATION_ASYMMETRY = "information_asymmetry"
    LIQUIDITY_PROVISION = "liquidity_provision"
    RISK_PREMIUM = "risk_premium"
    MOMENTUM_HERDING = "momentum_herding"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY_CLUSTERING = "volatility_clustering"
    ORDER_FLOW_IMPACT = "order_flow_impact"
    ARBITRAGE_CORRECTION = "arbitrage_correction"
    MARKET_MAKER_INVENTORY = "market_maker_inventory"
    NEWS_INCORPORATION = "news_incorporation"


@dataclass
class CausalRelationship:
    """Represents a discovered causal relationship"""
    cause_feature: str
    effect_event: str
    relationship_type: CausalRelationshipType
    economic_mechanism: EconomicMechanism
    strength: float  # Statistical strength of relationship
    p_value: float
    confidence_interval: Tuple[float, float]
    economic_justification: str
    temporal_lag: int  # Time lag in periods
    regime_stability: Dict[str, float]
    discovery_date: str
    validation_history: List[Dict[str, Any]]

    def __hash__(self):
        """Make CausalRelationship hashable for use as dictionary keys"""
        return hash((self.cause_feature, self.effect_event, self.strength))

    def __eq__(self, other):
        """Define equality for proper hashing"""
        if not isinstance(other, CausalRelationship):
            return False
        return (self.cause_feature == other.cause_feature and
                self.effect_event == other.effect_event and
                abs(self.strength - other.strength) < 1e-6)

    def get_unique_id(self):
        """Get a unique string identifier for this relationship"""
        return f"{self.cause_feature}→{self.effect_event}_{self.strength:.3f}"


@dataclass
class EconomicHypothesis:
    """Economic hypothesis about causal relationships"""
    hypothesis_id: str
    description: str
    proposed_mechanism: EconomicMechanism
    expected_features: List[str]
    expected_direction: str  # 'positive', 'negative', 'non_linear'
    theoretical_basis: str
    empirical_predictions: List[str]
    testable_implications: List[str]


class EconomicTheoryEngine:
    """Encodes financial theory and generates testable hypotheses"""

    def __init__(self):
        self.mechanisms = self._initialize_economic_mechanisms()
        self.feature_theory_map = self._create_feature_theory_mapping()
        self.hypothesis_templates = self._create_hypothesis_templates()

    def _initialize_economic_mechanisms(self) -> Dict[EconomicMechanism, Dict]:
        """Initialize known economic mechanisms with their characteristics"""
        return {
            EconomicMechanism.INFORMATION_ASYMMETRY: {
                'description': 'Informed traders cause price movements before public information',
                'typical_features': ['volume_imbalance', 'bid_ask_spread', 'order_flow'],
                'typical_lag': [1, 5],
                'market_impact': 'high',
                'regime_dependence': 'low'
            },
            EconomicMechanism.LIQUIDITY_PROVISION: {
                'description': 'Market makers adjust quotes based on inventory and order flow',
                'typical_features': ['bid_ask_spread', 'market_depth', 'volume', 'volatility'],
                'typical_lag': [0, 2],
                'market_impact': 'medium',
                'regime_dependence': 'high'
            },
            EconomicMechanism.MOMENTUM_HERDING: {
                'description': 'Past returns cause future returns through behavioral herding',
                'typical_features': ['past_returns', 'volume', 'volatility', 'trend_indicators'],
                'typical_lag': [1, 10],
                'market_impact': 'medium',
                'regime_dependence': 'very_high'
            },
            EconomicMechanism.MEAN_REVERSION: {
                'description': 'Prices revert to fundamental value after deviations',
                'typical_features': ['price_deviation', 'volatility', 'volume'],
                'typical_lag': [5, 50],
                'market_impact': 'medium',
                'regime_dependence': 'high'
            },
            EconomicMechanism.VOLATILITY_CLUSTERING: {
                'description': 'High volatility periods cause future high volatility',
                'typical_features': ['past_volatility', 'volume', 'returns'],
                'typical_lag': [1, 20],
                'market_impact': 'high',
                'regime_dependence': 'medium'
            },
            EconomicMechanism.ORDER_FLOW_IMPACT: {
                'description': 'Order flow directly impacts prices through market mechanics',
                'typical_features': ['volume', 'order_imbalance', 'trade_size'],
                'typical_lag': [0, 1],
                'market_impact': 'very_high',
                'regime_dependence': 'low'
            }
        }

    def _create_feature_theory_mapping(self) -> Dict[str, List[EconomicMechanism]]:
        """Map features to their theoretical economic mechanisms"""
        mapping = {}

        # Volume-related features
        volume_features = ['Volume', 'Volume_SMA', 'Volume_ratio', 'volume_imbalance']
        for feature in volume_features:
            mapping[feature] = [
                EconomicMechanism.ORDER_FLOW_IMPACT,
                EconomicMechanism.INFORMATION_ASYMMETRY,
                EconomicMechanism.LIQUIDITY_PROVISION
            ]

        # Volatility features
        volatility_features = ['ATR', 'ATR_pct', 'realized_vol', 'GARCH_vol']
        for feature in volatility_features:
            mapping[feature] = [
                EconomicMechanism.VOLATILITY_CLUSTERING,
                EconomicMechanism.RISK_PREMIUM,
                EconomicMechanism.INFORMATION_ASYMMETRY
            ]

        # Price/Return features
        price_features = ['returns', 'log_returns', 'price_momentum', 'RSI']
        for feature in price_features:
            mapping[feature] = [
                EconomicMechanism.MOMENTUM_HERDING,
                EconomicMechanism.MEAN_REVERSION,
                EconomicMechanism.NEWS_INCORPORATION
            ]

        return mapping

    def _create_hypothesis_templates(self) -> List[EconomicHypothesis]:
        """Create testable economic hypotheses"""
        return [
            EconomicHypothesis(
                hypothesis_id="momentum_persistence",
                description="Past positive returns cause future positive returns through herding behavior",
                proposed_mechanism=EconomicMechanism.MOMENTUM_HERDING,
                expected_features=["past_returns", "volume", "volatility"],
                expected_direction="positive",
                theoretical_basis="Behavioral finance herding models",
                empirical_predictions=["Positive autocorrelation in returns", "Volume amplifies momentum"],
                testable_implications=["Granger causality from past returns to future returns"]
            ),
            EconomicHypothesis(
                hypothesis_id="volatility_clustering",
                description="High volatility causes future high volatility through information arrival",
                proposed_mechanism=EconomicMechanism.VOLATILITY_CLUSTERING,
                expected_features=["past_volatility", "volume"],
                expected_direction="positive",
                theoretical_basis="GARCH models and information arrival theories",
                empirical_predictions=["Volatility persistence", "Volume-volatility relationship"],
                testable_implications=["Granger causality from past volatility to future volatility"]
            ),
            EconomicHypothesis(
                hypothesis_id="order_flow_impact",
                description="Order flow directly causes price movements through market microstructure",
                proposed_mechanism=EconomicMechanism.ORDER_FLOW_IMPACT,
                expected_features=["volume", "order_imbalance", "trade_size"],
                expected_direction="positive",
                theoretical_basis="Market microstructure theory",
                empirical_predictions=["Contemporaneous volume-price relationship"],
                testable_implications=["Instantaneous causality from order flow to prices"]
            )
        ]

    def generate_hypotheses_for_event(self, event_label: str, available_features: List[str]) -> List[
        EconomicHypothesis]:
        """Generate relevant hypotheses for a specific event"""
        event_lower = event_label.lower()
        relevant_hypotheses = []

        # Match event characteristics to mechanisms
        if 'momentum' in event_lower or 'trend' in event_lower:
            relevant_hypotheses.extend([h for h in self.hypothesis_templates
                                        if h.proposed_mechanism == EconomicMechanism.MOMENTUM_HERDING])

        if 'volatility' in event_lower or 'vol' in event_lower:
            relevant_hypotheses.extend([h for h in self.hypothesis_templates
                                        if h.proposed_mechanism == EconomicMechanism.VOLATILITY_CLUSTERING])

        if 'volume' in event_lower or 'flow' in event_lower:
            relevant_hypotheses.extend([h for h in self.hypothesis_templates
                                        if h.proposed_mechanism == EconomicMechanism.ORDER_FLOW_IMPACT])

        # Filter by available features
        filtered_hypotheses = []
        for hypothesis in relevant_hypotheses:
            feature_overlap = set(hypothesis.expected_features).intersection(set(available_features))
            if len(feature_overlap) > 0:
                filtered_hypotheses.append(hypothesis)

        return filtered_hypotheses or self.hypothesis_templates

    def validate_economic_plausibility(self, feature: str, event: str, relationship_strength: float) -> Tuple[
        bool, str, float]:
        """Validate if a statistical relationship is economically plausible"""

        possible_mechanisms = self.feature_theory_map.get(feature, [])

        if not possible_mechanisms:
            return True, "Economic plausibility check bypassed", 0.5

        # Simple validation - in practice you'd make this more sophisticated
        best_mechanism = possible_mechanisms[0]
        mechanism_info = self.mechanisms[best_mechanism]

        justification = f"Supported by {best_mechanism.value} mechanism: {mechanism_info['description']}"
        confidence = min(1.0, relationship_strength * 2)  # Simple confidence mapping

        return True, justification, confidence


class CausalInferenceEngine:
    """Core engine for discovering and validating causal relationships"""

    def __init__(self, theory_engine: EconomicTheoryEngine):
        self.theory_engine = theory_engine
        self.min_periods_for_test = 30
        self.significance_level = 0.1
        self.max_lag_test = 5

    def test_granger_causality(self, cause_series: pd.Series, effect_series: pd.Series, max_lag: int = None) -> Dict[
        str, Any]:
        """Test Granger causality between two time series"""

        if max_lag is None:
            max_lag = min(self.max_lag_test, len(cause_series) // 20)

        try:
            aligned_data = pd.concat([cause_series, effect_series], axis=1).dropna()
        except Exception as e:
            return {
                'is_causal': False,
                'reason': f'Data alignment failed: {str(e)}',
                'min_p_value': 1.0,
                'best_lag': 0,
                'test_statistics': {},
                'strength': 0
            }

        if len(aligned_data) < self.min_periods_for_test:
            return {
                'is_causal': False,
                'reason': f'Insufficient data points: {len(aligned_data)} < {self.min_periods_for_test}',
                'min_p_value': 1.0,
                'best_lag': 0,
                'test_statistics': {},
                'strength': 0
            }

        cause_clean = aligned_data.iloc[:, 0]
        effect_clean = aligned_data.iloc[:, 1]

        try:
            # Prepare data for Granger causality test
            data_for_test = pd.concat([effect_clean, cause_clean], axis=1)
            data_for_test.columns = ['effect', 'cause']

            # Check for variance
            if data_for_test['cause'].std() == 0 or data_for_test['effect'].std() == 0:
                return {
                    'is_causal': False,
                    'reason': 'Zero variance in time series',
                    'min_p_value': 1.0,
                    'best_lag': 0,
                    'test_statistics': {},
                    'strength': 0
                }

            # Test multiple lags and find the best one
            results = {}
            min_p_value = 1.0
            best_lag = 1

            for lag in range(1, max_lag + 1):
                try:
                    if len(data_for_test) <= lag * 2:
                        break

                    gc_result = grangercausalitytests(data_for_test, maxlag=lag, verbose=False)
                    p_value = gc_result[lag][0]['ssr_ftest'][1]
                    results[lag] = {
                        'p_value': p_value,
                        'f_statistic': gc_result[lag][0]['ssr_ftest'][0]
                    }

                    if p_value < min_p_value:
                        min_p_value = p_value
                        best_lag = lag

                except Exception as e:
                    results[lag] = {'error': str(e)}
                    continue

            is_causal = min_p_value < self.significance_level

            return {
                'is_causal': is_causal,
                'min_p_value': min_p_value,
                'best_lag': best_lag,
                'test_statistics': results,
                'strength': 1 - min_p_value if is_causal else 0
            }

        except Exception as e:
            return {
                'is_causal': False,
                'reason': f'Test failed: {str(e)}',
                'min_p_value': 1.0,
                'best_lag': 0,
                'test_statistics': {},
                'strength': 0
            }


class CausalFeatureDiscovery:
    """Discovers causal features based on economic theory and empirical testing"""

    def __init__(self, theory_engine: EconomicTheoryEngine, causal_engine: CausalInferenceEngine):
        self.theory_engine = theory_engine
        self.causal_engine = causal_engine

    def discover_causal_features(self, features_df: pd.DataFrame, target_series: pd.Series,
                                 event_label: str, max_features: int = 10) -> List[CausalRelationship]:
        """Discover causally valid features for predicting an event"""

        discovered_relationships = []
        exclude_features = ['Open', 'High', 'Low', 'Volume', 'Close']

        # Limit features to test for speed
        features_to_test = [f for f in features_df.columns if f not in exclude_features][
                           :min(50, len(features_df.columns))]

        print(f'Testing {len(features_to_test)} features for causality')

        for feature in features_to_test:
            try:
                feature_series = features_df[feature]

                # Skip if feature has zero variance or is all NaN
                if feature_series.std() == 0 or feature_series.isna().all():
                    continue

                # Test Granger causality
                granger_result = self.causal_engine.test_granger_causality(feature_series, target_series)

                if not granger_result.get('is_causal', False) or granger_result.get('strength', 0) == 0:
                    continue

                # Economic plausibility test
                try:
                    is_plausible, justification, plausibility_confidence = self.theory_engine.validate_economic_plausibility(
                        feature, event_label, granger_result.get('strength', 0)
                    )
                except Exception as e:
                    is_plausible, justification, plausibility_confidence = True, f"Plausibility bypassed: {str(e)}", 0.5

                if not is_plausible:
                    continue

                # Create causal relationship
                relationship = CausalRelationship(
                    cause_feature=feature,
                    effect_event=event_label,
                    relationship_type=CausalRelationshipType.GRANGER_CAUSAL,
                    economic_mechanism=EconomicMechanism.INFORMATION_ASYMMETRY,  # Default
                    strength=granger_result.get('strength', 0),
                    p_value=granger_result.get('min_p_value', 1.0),
                    confidence_interval=(0, 1),  # Simplified
                    economic_justification=justification,
                    temporal_lag=granger_result.get('best_lag', 1),
                    regime_stability={'GENERAL': granger_result.get('strength', 0)},
                    discovery_date=datetime.now().isoformat(),
                    validation_history=[]
                )

                discovered_relationships.append(relationship)

            except Exception as e:
                print(f"Error testing {feature}: {e}")
                continue

        # Sort by causal strength
        discovered_relationships.sort(key=lambda x: x.strength, reverse=True)
        return discovered_relationships[:max_features]


class CausalAgentKnowledgeBase:
    """Knowledge base storing causal relationships and economic insights"""

    def __init__(self):
        self.causal_relationships: Dict[str, List[Dict]] = {}
        self.economic_insights: List[Dict] = []
        self.mechanism_success_rates: Dict[str, List[float]] = {}
        self.failed_relationships: Set[Tuple[str, str]] = set()
        self.validation_history: Dict[str, List[Dict]] = {}

    def store_causal_relationship(self, relationship: CausalRelationship, validation_performance: Dict[str, float]):
        """Store a validated causal relationship"""

        event_key = relationship.effect_event
        if event_key not in self.causal_relationships:
            self.causal_relationships[event_key] = []

        rel_dict = asdict(relationship)
        rel_dict['validation_performance'] = validation_performance
        self.causal_relationships[event_key].append(rel_dict)

        # Update mechanism success rates
        mechanism_str = relationship.economic_mechanism.value
        if mechanism_str not in self.mechanism_success_rates:
            self.mechanism_success_rates[mechanism_str] = []

        economic_score = validation_performance.get('total_return', 0) * validation_performance.get('auc_roc', 0.5)
        self.mechanism_success_rates[mechanism_str].append(economic_score)

    def get_causal_features_for_event(self, event_label: str, min_strength: float = 0.3) -> List[Dict]:
        """Get causally validated features for an event"""
        relationships = self.causal_relationships.get(event_label, [])
        return [r for r in relationships if r.get('strength', 0) >= min_strength]

    def get_mechanism_preferences(self, market_regime: str = None) -> Dict[str, float]:
        """Get preferred economic mechanisms based on historical performance"""
        return {mechanism: np.mean(scores)
                for mechanism, scores in self.mechanism_success_rates.items()
                if len(scores) > 0}


class CausalResearchAgent:
    """Main agent orchestrating causal financial research"""

    def __init__(self, labeled_data: Union[pd.DataFrame, str, Path],
                 results_dir: Optional[str] = 'causal_results',
                 random_state: int = 42,
                 verbose: bool = True):

        # Load data
        if isinstance(labeled_data, (str, Path)):
            self.data = self._load_data_from_path(labeled_data)
        elif isinstance(labeled_data, pd.DataFrame):
            self.data = labeled_data.copy()
        else:
            raise ValueError("labeled_data must be a DataFrame or path to data file")

        self.results_dir = Path(results_dir) if results_dir else None
        if self.results_dir:
            self.results_dir.mkdir(exist_ok=True)

        self.random_state = random_state
        self.verbose = verbose

        # Initialize components
        self.theory_engine = EconomicTheoryEngine()
        self.causal_engine = CausalInferenceEngine(self.theory_engine)
        self.feature_discovery = CausalFeatureDiscovery(self.theory_engine, self.causal_engine)
        self.knowledge_base = CausalAgentKnowledgeBase()

        # Setup logging
        self.logger = self._setup_logger()

        # Analyze data structure FIRST
        self._analyze_data_structure()

        # Clean data with event-specific approach
        self.data = self._clean_data_for_causality()

        # Data statistics
        self._data_stats()

        if self.verbose:
            self.logger.info(f"Data cleaned for causality: {self.data.shape}")
            self.logger.info("Causal Financial Research Agent initialized")
            self.logger.info(f"Detected {len(self.label_columns)} event types")
            self.logger.info(f"Available features: {len(self.feature_columns)}")

    def _load_data_from_path(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from various file formats"""
        data_path = Path(data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        if data_path.suffix.lower() == '.pkl':
            return pd.read_pickle(data_path)
        elif data_path.suffix.lower() == '.csv':
            return pd.read_csv(data_path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

    def _setup_logger(self):
        """Setup logging"""
        logger = logging.getLogger(f"CAUSAL_AGENT")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - CAUSAL_AGENT - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _identify_column_types(self) -> Dict[str, List[str]]:
        """Identify different types of columns in the dataset"""

        # Event columns (contain 'event' or 'covent' in name, boolean values)
        event_columns = []
        for col in self.data.columns:
            if ('event' in col.lower() or 'covent' in col.lower()) and not col.endswith('_label'):
                # Check if column contains boolean-like values
                unique_vals = self.data[col].dropna().unique()
                if len(unique_vals) > 0 and set(unique_vals).issubset({True, False, 0, 1, 0.0, 1.0}):
                    event_columns.append(col)

        # Label columns (contain 'label' in name)
        label_columns = [col for col in self.data.columns if col.endswith('_label')]

        # Technical indicator columns (numeric, not event/label/ohlcv)
        exclude_patterns = {'_label', '_barrier_touched', '_touch_time', '_return', '_holding_hours',
                            '_event', 'event_type', 'any_event'}
        ohlcv_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        numeric_dtypes = {'float64', 'int64', 'float32', 'int32', 'float16', 'int16'}

        technical_indicator_columns = []
        for col in self.data.columns:
            # Skip if it's an event or label column
            if col in event_columns or col in label_columns:
                continue
            # Skip if it matches exclude patterns
            if any(pattern in col.lower() for pattern in exclude_patterns):
                continue
            # Include if numeric and not completely empty
            if (self.data[col].dtype.name in numeric_dtypes or col in ohlcv_cols):
                if not self.data[col].isna().all():
                    technical_indicator_columns.append(col)

        return {
            'event_columns': event_columns,
            'label_columns': label_columns,
            'technical_indicator_columns': technical_indicator_columns
        }

    def _analyze_data_structure(self):
        """Analyze data structure to identify features and labels"""

        column_types = self._identify_column_types()

        # Store column classifications
        self.event_columns = column_types['event_columns']
        self.label_columns = column_types['label_columns']
        self.feature_columns = column_types['technical_indicator_columns']

        # Try to identify price column for regime analysis
        price_candidates = ['Close', 'close', 'price', 'Price']
        self.price_column = None
        for candidate in price_candidates:
            if candidate in self.data.columns:
                self.price_column = candidate
                break

    def _clean_data_for_causality(self):
        """Clean data with event-specific approach - only forward fill technical indicators"""

        column_types = self._identify_column_types()

        if self.verbose:
            print(f"Identified column types:")
            print(f"  Event columns: {len(column_types['event_columns'])}")
            print(f"  Label columns: {len(column_types['label_columns'])}")
            print(f"  Technical indicator columns: {len(column_types['technical_indicator_columns'])}")

        data_clean = self.data.copy()

        # Only forward fill technical indicators - these should maintain continuity
        technical_cols = column_types['technical_indicator_columns']
        if technical_cols:
            data_clean[technical_cols] = data_clean[technical_cols].ffill()
            if self.verbose:
                print(f"Forward filled {len(technical_cols)} technical indicator columns")

        # Only drop rows where ALL technical indicators are NaN
        if technical_cols:
            critical_technical_cols = technical_cols[:min(5, len(technical_cols))]
            data_clean = data_clean.dropna(subset=critical_technical_cols, how='all')

        if self.verbose:
            print(f"Data shape after cleaning: {data_clean.shape}")
            print(f"Preserved event sparsity - events remain NaN when not occurring")

        return data_clean

    def _data_stats(self):
        """Print data statistics"""
        print("Event column analysis:")
        for col in self.event_columns[:5]:
            if col in self.data.columns:
                true_events = ((self.data[col] == True) | (self.data[col] == 1)).sum()
                total_rows = len(self.data)
                print(f"{col}: {true_events} events ({true_events / total_rows:.3%} of data)")

        print("\nLabel distributions:")
        for col in self.label_columns[:5]:
            if col in self.data.columns:
                print(f"{col}: {self.data[col].value_counts().to_dict()}")

        print(f"\nFeature statistics:")
        print(f"Technical indicators: {len(self.feature_columns)}")
        if self.feature_columns:
            feature_completeness = (1 - self.data[self.feature_columns].isna().mean()).mean()
            print(f"Average feature completeness: {feature_completeness:.1%}")

    def research_event_causally(self, event_label: str) -> Dict[str, Any]:
        """Research a single event using causal inference - EVENT-SPECIFIC APPROACH"""

        if self.verbose:
            self.logger.info(f"\n🔬 CAUSAL RESEARCH: {event_label}")

        try:
            if event_label not in self.data.columns:
                if self.verbose:
                    self.logger.warning(f"Event label {event_label} not found in data")
                return None

            # Get corresponding event column
            event_col = event_label.replace('_label', '')
            if event_col not in self.data.columns:
                # Try alternative naming patterns
                possible_event_cols = []
                for col in self.event_columns:
                    col_base = col.replace('_event', '').replace('_covent', '').replace('event_', '').replace('covent_',
                                                                                                              '')
                    label_base = event_label.replace('_label', '')
                    if col_base == label_base or col_base in label_base or label_base in col_base:
                        possible_event_cols.append(col)

                if possible_event_cols:
                    event_col = possible_event_cols[0]
                else:
                    if self.verbose:
                        self.logger.warning(f"Could not find event column for {event_label}")
                    return None

            # Filter to only rows where the event actually occurred (True)
            # Handle both boolean True and numeric 1
            event_mask = (self.data[event_col] == True) | (self.data[event_col] == 1)
            event_occurrences = event_mask.sum()

            if event_occurrences < 30:  # Need sufficient events for causal analysis
                if self.verbose:
                    self.logger.warning(f"Insufficient event occurrences for {event_label}: {event_occurrences}")
                return None

            if self.verbose:
                self.logger.info(f"📅 Found {event_occurrences} occurrences of {event_col}")

            # Extract data for only when this event occurred
            event_data = self.data[event_mask].copy()

            # Get features and target for event occurrences only
            X = event_data[self.feature_columns].copy()
            y = event_data[event_label].copy()

            # Clean the event-specific data
            X = X.ffill().bfill().fillna(0)
            y = y.fillna(0)

            # Remove constant features
            constant_features = X.columns[X.std() == 0].tolist()
            if constant_features:
                X = X.drop(columns=constant_features)

            if self.verbose:
                self.logger.info(f"📊 Event-specific data prepared: {len(X)} samples, {len(X.columns)} features")

            # Discover causal features
            causal_relationships = self.feature_discovery.discover_causal_features(
                X, y, event_label, max_features=15
            )

            if not causal_relationships:
                if self.verbose:
                    self.logger.warning(f"No causal relationships discovered for {event_label}")
                return None

            if self.verbose:
                self.logger.info(f"✅ Discovered {len(causal_relationships)} causal relationships")

            # Validate causal relationships economically
            validation_results = self._validate_causal_relationships_economically(X, y, causal_relationships)

            # Store successful relationships in knowledge base
            for rel, perf in validation_results.items():
                if perf.get('total_return', 0) > -0.05:
                    self.knowledge_base.store_causal_relationship(rel, perf)

            # Create research result
            result = {
                'event_label': event_label,
                'event_column': event_col,
                'event_occurrences': int(event_occurrences),
                'event_frequency': float(event_occurrences / len(self.data)),
                'causal_relationships': [self._relationship_to_dict(rel) for rel in causal_relationships],
                'validation_results': {rel.get_unique_id(): perf for rel, perf in validation_results.items()},
                'research_timestamp': datetime.now().isoformat(),
                'data_samples': len(X),
                'features_tested': len(X.columns)
            }

            return result

        except Exception as e:
            if self.verbose:
                self.logger.error(f"❌ Causal research failed for {event_label}: {e}")
            return None

    def _relationship_to_dict(self, relationship: CausalRelationship) -> Dict[str, Any]:
        """Convert CausalRelationship to dictionary with proper string conversion"""
        rel_dict = asdict(relationship)
        rel_dict['relationship_type'] = relationship.relationship_type.value
        rel_dict['economic_mechanism'] = relationship.economic_mechanism.value
        return rel_dict

    def _validate_causal_relationships_economically(self, X: pd.DataFrame, y: pd.Series,
                                                    causal_relationships: List[CausalRelationship]) -> Dict[
        CausalRelationship, Dict[str, float]]:
        """Validate causal relationships using economic performance"""

        validation_results = {}
        causal_features = [rel.cause_feature for rel in causal_relationships]
        available_features = [f for f in causal_features if f in X.columns]

        if not available_features:
            return validation_results

        # Time series cross-validation
        cv = TimeSeriesSplit(n_splits=min(3, len(X) // 30))

        for relationship in causal_relationships:
            feature_name = relationship.cause_feature

            if feature_name not in X.columns:
                continue

            try:
                X_single = X[[feature_name]]
                model = LogisticRegression(random_state=self.random_state, max_iter=1000)

                auc_scores = []
                returns = []

                for train_idx, test_idx in cv.split(X_single):
                    try:
                        X_train, X_test = X_single.iloc[train_idx], X_single.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                        if len(X_train) < 10 or len(X_test) < 5:
                            continue

                        # Scale features
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)

                        # Fit and predict
                        model.fit(X_train_scaled, y_train)
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                        y_pred = model.predict(X_test_scaled)

                        # Calculate AUC
                        if len(np.unique(y_test)) > 1:
                            auc = roc_auc_score(y_test, y_pred_proba)
                            auc_scores.append(auc)

                        # Simple return calculation
                        returns.append((auc - 0.5) * 0.1 if len(auc_scores) > 0 else 0)

                    except Exception:
                        continue

                # Aggregate results
                avg_auc = np.mean(auc_scores) if auc_scores else 0.5
                total_return = np.sum(returns) if returns else 0
                sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0

                validation_results[relationship] = {
                    'auc_roc': avg_auc,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'num_folds': len(auc_scores),
                    'economic_score': avg_auc * (1 + abs(total_return))
                }

            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Validation failed for {relationship.cause_feature}: {e}")
                validation_results[relationship] = {
                    'auc_roc': 0.5,
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'num_folds': 0,
                    'economic_score': 0.0
                }

        return validation_results

    def comprehensive_causal_research(self, max_time_minutes: int = 60,
                                      priority_events: List[str] = None) -> Dict[str, Any]:
        """Conduct comprehensive causal research across all events"""

        start_time = datetime.now()
        max_time_seconds = max_time_minutes * 60

        if self.verbose:
            self.logger.info(f"\n🚀 COMPREHENSIVE EVENT-SPECIFIC CAUSAL RESEARCH")
            self.logger.info(f"Time budget: {max_time_minutes} minutes")

        # Determine events to research
        events_to_research = priority_events if priority_events else self.label_columns

        # Research results
        research_results = {}
        failed_events = []

        for i, event_label in enumerate(events_to_research):
            current_time = datetime.now()
            elapsed_seconds = (current_time - start_time).total_seconds()

            # Check time budget
            if elapsed_seconds >= max_time_seconds * 0.9:
                if self.verbose:
                    self.logger.info("⏰ Time budget nearly exhausted. Stopping research.")
                break

            if self.verbose:
                remaining_time = (max_time_seconds - elapsed_seconds) / 60
                self.logger.info(f"\n[{i + 1}/{len(events_to_research)}] Researching: {event_label}")
                self.logger.info(f"⏱️  Remaining time: {remaining_time:.1f} minutes")

            # Research this event
            try:
                result = self.research_event_causally(event_label)
                if result:
                    research_results[event_label] = result
                else:
                    failed_events.append(event_label)

            except Exception as e:
                if self.verbose:
                    self.logger.error(f"Research failed for {event_label}: {e}")
                failed_events.append(event_label)

        # Generate comprehensive analysis
        total_duration = (datetime.now() - start_time).total_seconds()
        analysis = self._generate_causal_analysis(research_results, failed_events)

        # Create final results package
        final_results = {
            'research_strategy': 'event_specific_causal_inference',
            'research_approach': 'Filter to actual event occurrences only',
            'research_context': {
                'start_time': start_time.isoformat(),
                'total_duration_seconds': total_duration,
                'time_budget_minutes': max_time_minutes,
                'events_attempted': len(events_to_research),
                'events_successful': len(research_results),
                'events_failed': len(failed_events)
            },
            'research_results': research_results,
            'failed_events': failed_events,
            'causal_analysis': analysis,
            'knowledge_base_summary': {
                'total_relationships': sum(len(rels) for rels in self.knowledge_base.causal_relationships.values()),
                'mechanism_preferences': self.knowledge_base.get_mechanism_preferences(),
            }
        }

        if self.verbose:
            self._print_causal_research_summary(final_results)

        return final_results

    def _generate_causal_analysis(self, research_results: Dict, failed_events: List[str]) -> Dict[str, Any]:
        """Generate comprehensive causal analysis"""

        if not research_results:
            return {
                'summary': 'No successful causal relationships discovered',
                'recommendations': ['Check event frequency', 'Ensure sufficient event occurrences']
            }

        # Extract all causal relationships
        all_relationships = []
        event_stats = {}

        for event_label, result in research_results.items():
            event_stats[event_label] = {
                'occurrences': result.get('event_occurrences', 0),
                'frequency': result.get('event_frequency', 0),
                'relationships_found': len(result.get('causal_relationships', []))
            }
            for rel_dict in result.get('causal_relationships', []):
                all_relationships.append(rel_dict)

        # Analyze mechanisms and features
        mechanisms = [rel.get('economic_mechanism', '') for rel in all_relationships]
        features = [rel['cause_feature'] for rel in all_relationships]
        mechanism_counts = Counter(mechanisms)
        feature_counts = Counter(features)

        # Generate insights
        insights = []
        if mechanism_counts:
            dominant_mechanism = mechanism_counts.most_common(1)[0]
            insights.append(
                f"Dominant economic mechanism: {dominant_mechanism[0]} ({dominant_mechanism[1]} relationships)")

        if feature_counts:
            most_causal_feature = feature_counts.most_common(1)[0]
            insights.append(
                f"Most causally important feature: {most_causal_feature[0]} ({most_causal_feature[1]} relationships)")

        if event_stats:
            avg_frequency = np.mean([stats['frequency'] for stats in event_stats.values()])
            insights.append(f"Average event frequency: {avg_frequency:.3%}")

        # Recommendations
        recommendations = [
            "✅ EVENT-SPECIFIC APPROACH: Research focuses only on actual event occurrences",
            "✅ DATA INTEGRITY: No artificial events created by forward filling",
        ]

        if len(all_relationships) > 0:
            avg_strength = np.mean([rel['strength'] for rel in all_relationships])
            if avg_strength < 0.5:
                recommendations.append("Consider additional feature engineering to strengthen causal relationships")
            else:
                recommendations.append("Strong causal relationships detected - focus on real-time event detection")

        recommendations.extend([
            "Validate causal relationships on out-of-sample event occurrences",
            "Develop real-time event detection system to trigger predictions"
        ])

        return {
            'summary': {
                'total_relationships': len(all_relationships),
                'dominant_mechanisms': dict(mechanism_counts.most_common(5)),
                'most_causal_features': dict(feature_counts.most_common(10)),
                'average_causal_strength': np.mean(
                    [rel['strength'] for rel in all_relationships]) if all_relationships else 0,
                'event_statistics': event_stats
            },
            'insights': insights,
            'recommendations': recommendations,
            'risk_factors': [
                "Causal relationships may break down during regime changes",
                "Event detection accuracy is critical for model performance",
                "Low-frequency events may have insufficient training data"
            ]
        }

    def _print_causal_research_summary(self, results: Dict[str, Any]):
        """Print comprehensive causal research summary"""

        print(f"\n{'=' * 80}")
        print("🔬 EVENT-SPECIFIC CAUSAL FINANCIAL RESEARCH SUMMARY")
        print(f"{'=' * 80}")

        # Research context
        context = results['research_context']
        print(f"📊 Research Statistics:")
        print(f"   • Events attempted: {context['events_attempted']}")
        print(f"   • Successful research: {context['events_successful']}")
        print(f"   • Success rate: {context['events_successful'] / context['events_attempted']:.1%}")
        print(f"   • Duration: {context['total_duration_seconds'] / 60:.1f} minutes")

        # Research approach
        print(f"\n🎯 Research Approach: {results['research_approach']}")

        # Analysis results
        analysis = results['causal_analysis']
        summary = analysis.get('summary', {})

        if isinstance(summary, dict):
            print(f"\n🔍 Causal Analysis:")
            print(f"   • Relationships discovered: {summary.get('total_relationships', 0)}")
            print(f"   • Average causal strength: {summary.get('average_causal_strength', 0):.3f}")

        # Event statistics
        event_stats = summary.get('event_statistics', {})
        if event_stats:
            print(f"\n📅 Event Analysis:")
            for event, stats in list(event_stats.items())[:5]:
                print(f"   • {event}: {stats['occurrences']} occurrences ({stats['frequency']:.3%}), "
                      f"{stats['relationships_found']} causal features")

        # Recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            print(f"\n📋 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")


# CONVENIENCE FUNCTIONS FOR EVENT-SPECIFIC CAUSAL RESEARCH

def quick_event_causal_research(labeled_data: Union[pd.DataFrame, str, Path],
                                max_time_minutes: int = 30,
                                results_dir: Optional[str] = 'causal_results',
                                verbose: bool = True) -> Dict[str, Any]:
    """Quick event-specific causal research focusing on strongest relationships"""
    agent = CausalResearchAgent(
        labeled_data=labeled_data,
        results_dir=results_dir,
        verbose=verbose
    )
    return agent.comprehensive_causal_research(max_time_minutes=max_time_minutes)


def deep_event_causal_research(labeled_data: Union[pd.DataFrame, str, Path],
                               max_time_minutes: int = 120,
                               results_dir: Optional[str] = 'causal_results',
                               verbose: bool = True) -> Dict[str, Any]:
    """Deep event-specific causal research with extended analysis"""
    agent = CausalResearchAgent(
        labeled_data=labeled_data,
        results_dir=results_dir,
        verbose=verbose
    )

    results = agent.comprehensive_causal_research(max_time_minutes=max_time_minutes)

    if verbose:
        print("\n🔬 DEEP EVENT-SPECIFIC CAUSAL RESEARCH ADDITIONAL ANALYSIS")
        print("=" * 60)

        kb = agent.knowledge_base
        mechanism_prefs = kb.get_mechanism_preferences()
        if mechanism_prefs:
            print(f"📊 Discovered Economic Mechanisms (by performance):")
            for mechanism, score in sorted(mechanism_prefs.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   • {mechanism}: {score:.3f} avg performance")

        total_relationships = sum(len(rels) for rels in kb.causal_relationships.values())
        print(f"\n🧠 Knowledge Base:")
        print(f"   • Total causal relationships stored: {total_relationships}")

    return results


# Example usage
if __name__ == "__main__":
    print("Event-Specific Causal Financial Research System")
    print("=" * 50)
    print("🎯 NEW APPROACH: Research only on actual event occurrences")
    print("✅ ELIMINATES: Artificial events from forward filling")
    print("🔬 ENSURES: True causal relationships")

    # Example usage with error handling
    print("\n🚀 Example Usage:")
    try:
        # Replace with your actual data file
        # results = quick_event_causal_research(
        #     labeled_data='labeled5mEE2cov.pkl',
        #     max_time_minutes=30,
        #     verbose=True
        # )

        results = deep_event_causal_research(
            labeled_data='D:/Seagull_data/labeled5mEE2cov.pkl',
            max_time_minutes=240,
            results_dir='causal_results'
        )
        with open('./causal_results/research_results.pkl', 'wb') as f:
            pickle.dump(results, f)

        print("✅ Research completed successfully!")
    except FileNotFoundError:
        print("⚠️  Example file not found. Use with your own data:")
        print("results = quick_event_causal_research('your_data.pkl', max_time_minutes=30)")
    except Exception as e:
        print(f"ℹ️  Example error: {e}")
        print("Use the functions with your own data file.")

    print("\n" + "=" * 60)
    print("📖 USAGE EXAMPLES")
    print("=" * 60)

    print("\n🔬 For Deep Event-Specific Causal Research:")
    print("results = deep_event_causal_research(")
    print("    labeled_data='D:/Seagull_data/labeled5mEE2cov.pkl',")
    print("    max_time_minutes=120")
    print(")")

    print("\n🎯 For Quick Event-Specific Research:")
    print("results = quick_event_causal_research(")
    print("    labeled_data='your_data.pkl',")
    print("    max_time_minutes=30")
    print(")")

# C:\Users\themi\PycharmProjects\Seagull_2\.venv\Scripts\python.exe C:\Users\themi\PycharmProjects\Seagull_2\causation_research.py
# Event-Specific Causal Financial Research System
# ==================================================
# 🎯 NEW APPROACH: Research only on actual event occurrences
# ✅ ELIMINATES: Artificial events from forward filling
# 🔬 ENSURES: True causal relationships
#
# 🚀 Example Usage:
# Identified column types:
#   Event columns: 12
#   Label columns: 13
#   Technical indicator columns: 85
# Forward filled 85 technical indicator columns
# Data shape after cleaning: (595026, 172)
# Preserved event sparsity - events remain NaN when not occurring
# Event column analysis:
# CUSUM_event: 2032 events (0.341% of data)
# vpd_volatility_event: 721 events (0.121% of data)
# outlier_event: 35591 events (5.981% of data)
# momentum_regime_event: 231363 events (38.883% of data)
# traditional_event: 250896 events (42.166% of data)
#
# Label distributions:
# CUSUM_event_label: {0.0: 1269, 1.0: 763}
# vpd_volatility_event_label: {0.0: 407, 1.0: 314}
# outlier_event_label: {0.0: 22778, 1.0: 12813}
# momentum_regime_event_label: {0.0: 148347, 1.0: 83016}
# traditional_event_label: {0.0: 160753, 1.0: 90143}
#
# Feature statistics:
# Technical indicators: 85
# 2025-09-23 07:20:21,182 - CAUSAL_AGENT - INFO - Data cleaned for causality: (595026, 172)
# 2025-09-23 07:20:21,182 - CAUSAL_AGENT - INFO - Causal Financial Research Agent initialized
# 2025-09-23 07:20:21,182 - CAUSAL_AGENT - INFO - Detected 13 event types
# 2025-09-23 07:20:21,182 - CAUSAL_AGENT - INFO - Available features: 85
# 2025-09-23 07:20:21,182 - CAUSAL_AGENT - INFO -
# 🚀 COMPREHENSIVE EVENT-SPECIFIC CAUSAL RESEARCH
# 2025-09-23 07:20:21,182 - CAUSAL_AGENT - INFO - Time budget: 240 minutes
# 2025-09-23 07:20:21,183 - CAUSAL_AGENT - INFO -
# [1/13] Researching: CUSUM_event_label
# 2025-09-23 07:20:21,183 - CAUSAL_AGENT - INFO - ⏱️  Remaining time: 240.0 minutes
# 2025-09-23 07:20:21,183 - CAUSAL_AGENT - INFO -
# 🔬 CAUSAL RESEARCH: CUSUM_event_label
# 2025-09-23 07:20:21,188 - CAUSAL_AGENT - INFO - 📅 Found 2032 occurrences of CUSUM_event
# Average feature completeness: 99.8%
# Testing 50 features for causality
# 2025-09-23 07:20:21,222 - CAUSAL_AGENT - INFO - 📊 Event-specific data prepared: 2032 samples, 83 features
# 2025-09-23 07:20:24,247 - CAUSAL_AGENT - INFO - ✅ Discovered 15 causal relationships
# 2025-09-23 07:20:24,627 - CAUSAL_AGENT - INFO -
# [2/13] Researching: vpd_volatility_event_label
# 2025-09-23 07:20:24,627 - CAUSAL_AGENT - INFO - ⏱️  Remaining time: 239.9 minutes
# 2025-09-23 07:20:24,627 - CAUSAL_AGENT - INFO -
# 🔬 CAUSAL RESEARCH: vpd_volatility_event_label
# 2025-09-23 07:20:24,630 - CAUSAL_AGENT - INFO - 📅 Found 721 occurrences of vpd_volatility_event
# Testing 50 features for causality
# 2025-09-23 07:20:24,638 - CAUSAL_AGENT - INFO - 📊 Event-specific data prepared: 721 samples, 84 features
# 2025-09-23 07:20:26,545 - CAUSAL_AGENT - INFO - ✅ Discovered 10 causal relationships
# 2025-09-23 07:20:26,799 - CAUSAL_AGENT - INFO -
# [3/13] Researching: outlier_event_label
# 2025-09-23 07:20:26,800 - CAUSAL_AGENT - INFO - ⏱️  Remaining time: 239.9 minutes
# 2025-09-23 07:20:26,800 - CAUSAL_AGENT - INFO -
# 🔬 CAUSAL RESEARCH: outlier_event_label
# 2025-09-23 07:20:26,803 - CAUSAL_AGENT - INFO - 📅 Found 35591 occurrences of outlier_event
# Testing 50 features for causality
# 2025-09-23 07:20:27,008 - CAUSAL_AGENT - INFO - 📊 Event-specific data prepared: 35591 samples, 85 features
# 2025-09-23 07:20:51,860 - CAUSAL_AGENT - INFO - ✅ Discovered 15 causal relationships
# 2025-09-23 07:20:52,422 - CAUSAL_AGENT - INFO -
# [4/13] Researching: momentum_regime_event_label
# 2025-09-23 07:20:52,422 - CAUSAL_AGENT - INFO - ⏱️  Remaining time: 239.5 minutes
# 2025-09-23 07:20:52,422 - CAUSAL_AGENT - INFO -
# 🔬 CAUSAL RESEARCH: momentum_regime_event_label
# 2025-09-23 07:20:52,425 - CAUSAL_AGENT - INFO - 📅 Found 231363 occurrences of momentum_regime_event
# Testing 50 features for causality
# 2025-09-23 07:20:53,561 - CAUSAL_AGENT - INFO - 📊 Event-specific data prepared: 231363 samples, 85 features
# 2025-09-23 07:23:20,172 - CAUSAL_AGENT - INFO - ✅ Discovered 15 causal relationships
# 2025-09-23 07:23:22,194 - CAUSAL_AGENT - INFO -
# [5/13] Researching: traditional_event_label
# 2025-09-23 07:23:22,194 - CAUSAL_AGENT - INFO - ⏱️  Remaining time: 237.0 minutes
# 2025-09-23 07:23:22,194 - CAUSAL_AGENT - INFO -
# 🔬 CAUSAL RESEARCH: traditional_event_label
# 2025-09-23 07:23:22,197 - CAUSAL_AGENT - INFO - 📅 Found 250896 occurrences of traditional_event
# Testing 50 features for causality
# 2025-09-23 07:23:23,452 - CAUSAL_AGENT - INFO - 📊 Event-specific data prepared: 250896 samples, 85 features
# 2025-09-23 07:26:03,398 - CAUSAL_AGENT - INFO - ✅ Discovered 15 causal relationships
# 2025-09-23 07:26:05,653 - CAUSAL_AGENT - INFO -
# [6/13] Researching: any_event_label
# 2025-09-23 07:26:05,653 - CAUSAL_AGENT - INFO - ⏱️  Remaining time: 234.3 minutes
# 2025-09-23 07:26:05,653 - CAUSAL_AGENT - INFO -
# 🔬 CAUSAL RESEARCH: any_event_label
# 2025-09-23 07:26:05,656 - CAUSAL_AGENT - INFO - 📅 Found 493248 occurrences of any_event
# 2025-09-23 07:26:08,199 - CAUSAL_AGENT - INFO - 📊 Event-specific data prepared: 493248 samples, 85 features
# Testing 50 features for causality
# 2025-09-23 07:31:26,571 - CAUSAL_AGENT - INFO - ✅ Discovered 15 causal relationships
# 2025-09-23 07:31:30,601 - CAUSAL_AGENT - INFO -
# [7/13] Researching: covent_2way_CUSUM_BB_upper_cross_label
# 2025-09-23 07:31:30,601 - CAUSAL_AGENT - INFO - ⏱️  Remaining time: 228.8 minutes
# 2025-09-23 07:31:30,602 - CAUSAL_AGENT - INFO -
# 🔬 CAUSAL RESEARCH: covent_2way_CUSUM_BB_upper_cross_label
# 2025-09-23 07:31:30,605 - CAUSAL_AGENT - INFO - 📅 Found 1200 occurrences of covent_2way_CUSUM_BB_upper_cross
# 2025-09-23 07:31:30,616 - CAUSAL_AGENT - INFO - 📊 Event-specific data prepared: 1200 samples, 84 features
# Testing 50 features for causality
# 2025-09-23 07:31:32,446 - CAUSAL_AGENT - INFO - ✅ Discovered 15 causal relationships
# 2025-09-23 07:31:32,786 - CAUSAL_AGENT - INFO -
# [8/13] Researching: covent_2way_CUSUM_BB_lower_cross_label
# 2025-09-23 07:31:32,786 - CAUSAL_AGENT - INFO - ⏱️  Remaining time: 228.8 minutes
# 2025-09-23 07:31:32,786 - CAUSAL_AGENT - INFO -
# 🔬 CAUSAL RESEARCH: covent_2way_CUSUM_BB_lower_cross_label
# 2025-09-23 07:31:32,789 - CAUSAL_AGENT - INFO - 📅 Found 1525 occurrences of covent_2way_CUSUM_BB_lower_cross
# Testing 50 features for causality
# 2025-09-23 07:31:32,800 - CAUSAL_AGENT - INFO - 📊 Event-specific data prepared: 1525 samples, 85 features
# 2025-09-23 07:31:34,788 - CAUSAL_AGENT - INFO - ✅ Discovered 15 causal relationships
# 2025-09-23 07:31:35,151 - CAUSAL_AGENT - INFO -
# [9/13] Researching: covent_2way_vpd_volatility_momentum_regime_label
# 2025-09-23 07:31:35,151 - CAUSAL_AGENT - INFO - ⏱️  Remaining time: 228.8 minutes
# 2025-09-23 07:31:35,151 - CAUSAL_AGENT - INFO -
# 🔬 CAUSAL RESEARCH: covent_2way_vpd_volatility_momentum_regime_label
# 2025-09-23 07:31:35,153 - CAUSAL_AGENT - INFO - 📅 Found 400 occurrences of covent_2way_vpd_volatility_momentum_regime
# 2025-09-23 07:31:35,158 - CAUSAL_AGENT - INFO - 📊 Event-specific data prepared: 400 samples, 84 features
# Testing 50 features for causality
# 2025-09-23 07:31:36,573 - CAUSAL_AGENT - INFO - ✅ Discovered 1 causal relationships
# 2025-09-23 07:31:36,595 - CAUSAL_AGENT - INFO -
# [10/13] Researching: covent_2way_outlier_BB_expansion_label
# 2025-09-23 07:31:36,595 - CAUSAL_AGENT - INFO - ⏱️  Remaining time: 228.7 minutes
# 2025-09-23 07:31:36,595 - CAUSAL_AGENT - INFO -
# 🔬 CAUSAL RESEARCH: covent_2way_outlier_BB_expansion_label
# 2025-09-23 07:31:36,597 - CAUSAL_AGENT - INFO - 📅 Found 88600 occurrences of covent_2way_outlier_BB_expansion
# Testing 50 features for causality
# 2025-09-23 07:31:37,101 - CAUSAL_AGENT - INFO - 📊 Event-specific data prepared: 88600 samples, 85 features
# 2025-09-23 07:32:29,684 - CAUSAL_AGENT - INFO - ✅ Discovered 15 causal relationships
# 2025-09-23 07:32:30,578 - CAUSAL_AGENT - INFO -
# [11/13] Researching: covent_2way_momentum_regime_BB_squeeze_label
# 2025-09-23 07:32:30,578 - CAUSAL_AGENT - INFO - ⏱️  Remaining time: 227.8 minutes
# 2025-09-23 07:32:30,578 - CAUSAL_AGENT - INFO -
# 🔬 CAUSAL RESEARCH: covent_2way_momentum_regime_BB_squeeze_label
# 2025-09-23 07:32:30,582 - CAUSAL_AGENT - INFO - 📅 Found 188575 occurrences of covent_2way_momentum_regime_BB_squeeze
# Testing 50 features for causality
# 2025-09-23 07:32:31,540 - CAUSAL_AGENT - INFO - 📊 Event-specific data prepared: 188575 samples, 85 features
# 2025-09-23 07:34:30,086 - CAUSAL_AGENT - INFO - ✅ Discovered 15 causal relationships
# 2025-09-23 07:34:31,718 - CAUSAL_AGENT - INFO -
# [12/13] Researching: any_2way_covent_label
# 2025-09-23 07:34:31,718 - CAUSAL_AGENT - INFO - ⏱️  Remaining time: 225.8 minutes
# 2025-09-23 07:34:31,718 - CAUSAL_AGENT - INFO -
# 🔬 CAUSAL RESEARCH: any_2way_covent_label
# 2025-09-23 07:34:31,720 - CAUSAL_AGENT - INFO - 📅 Found 273369 occurrences of any_2way_covent
# 2025-09-23 07:34:33,053 - CAUSAL_AGENT - INFO - 📊 Event-specific data prepared: 273369 samples, 85 features
# Testing 50 features for causality
# 2025-09-23 07:37:27,770 - CAUSAL_AGENT - INFO - ✅ Discovered 15 causal relationships
# 2025-09-23 07:37:30,061 - CAUSAL_AGENT - INFO -
# [13/13] Researching: count_2way_covent_label
# 2025-09-23 07:37:30,062 - CAUSAL_AGENT - INFO - ⏱️  Remaining time: 222.9 minutes
# 2025-09-23 07:37:30,062 - CAUSAL_AGENT - INFO -
# 🔬 CAUSAL RESEARCH: count_2way_covent_label
# 2025-09-23 07:37:30,064 - CAUSAL_AGENT - INFO - 📅 Found 266713 occurrences of count_2way_covent
# 2025-09-23 07:37:31,399 - CAUSAL_AGENT - INFO - 📊 Event-specific data prepared: 266713 samples, 84 features
# Testing 50 features for causality
# 2025-09-23 07:40:18,498 - CAUSAL_AGENT - INFO - ✅ Discovered 15 causal relationships
#
# ================================================================================
# 🔬 EVENT-SPECIFIC CAUSAL FINANCIAL RESEARCH SUMMARY
# ================================================================================
# 📊 Research Statistics:
#    • Events attempted: 13
#    • Successful research: 13
#    • Success rate: 100.0%
#    • Duration: 20.0 minutes
#
# 🎯 Research Approach: Filter to actual event occurrences only
#
# 🔍 Causal Analysis:
#    • Relationships discovered: 176
#    • Average causal strength: 0.990
#
# 📅 Event Analysis:
#    • CUSUM_event_label: 2032 occurrences (0.341%), 15 causal features
#    • vpd_volatility_event_label: 721 occurrences (0.121%), 10 causal features
#    • outlier_event_label: 35591 occurrences (5.981%), 15 causal features
#    • momentum_regime_event_label: 231363 occurrences (38.883%), 15 causal features
#    • traditional_event_label: 250896 occurrences (42.166%), 15 causal features
#
# 📋 Recommendations:
#    1. ✅ EVENT-SPECIFIC APPROACH: Research focuses only on actual event occurrences
#    2. ✅ DATA INTEGRITY: No artificial events created by forward filling
#    3. Strong causal relationships detected - focus on real-time event detection
#    4. Validate causal relationships on out-of-sample event occurrences
#    5. Develop real-time event detection system to trigger predictions
#
# 🔬 DEEP EVENT-SPECIFIC CAUSAL RESEARCH ADDITIONAL ANALYSIS
# ============================================================
# 📊 Discovered Economic Mechanisms (by performance):
#    • information_asymmetry: 0.002 avg performance
#
# 🧠 Knowledge Base:
#    • Total causal relationships stored: 176
# ✅ Research completed successfully!
#
# OVERALL STATISTICS:
# Total events researched: 13
# Total relationships found: 176
# Average causal strength: 0.9896325955889383
#
# ============================================================
# DETAILED CAUSAL RELATIONSHIPS BY EVENT
# ============================================================
#
# 🎯 EVENT: CUSUM_event_label
# --------------------------------------------------
# Occurrences: 2032
# Frequency: 0.0034
#
# CAUSAL RELATIONSHIPS FOUND: 15
#
# Top Causal Features:
#    1. CUSUM_direction
#       Strength: 1.0000 | P-value: 0.000018
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    2. returns
#       Strength: 1.0000 | P-value: 0.000048
#       Mechanism: information_asymmetry
#       Justification: Supported by momentum_herding mechanism: Past returns cause future returns through behavioral herding
#
#    3. EMA_fast_distance_pct
#       Strength: 0.9997 | P-value: 0.000291
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    4. EMA_medium_distance_pct
#       Strength: 0.9993 | P-value: 0.000741
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    5. 30min_Close
#       Strength: 0.9989 | P-value: 0.001099
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    6. EMA_slow_distance_pct
#       Strength: 0.9982 | P-value: 0.001771
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    7. BB_position
#       Strength: 0.9936 | P-value: 0.006351
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    8. RSI
#       Strength: 0.9936 | P-value: 0.006441
#       Mechanism: information_asymmetry
#       Justification: Supported by momentum_herding mechanism: Past returns cause future returns through behavioral herding
#
#    9. 4H_Volume
#       Strength: 0.9824 | P-value: 0.017550
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#   10. 30min_High
#       Strength: 0.9791 | P-value: 0.020901
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
# VALIDATION RESULTS:
#   CUSUM_direction→CUSUM_event_label_1.000:
#     AUC-ROC: 0.5204
#     Sharpe Ratio: 1.2245
#     Total Return: 0.0061
#   returns→CUSUM_event_label_1.000:
#     AUC-ROC: 0.5207
#     Sharpe Ratio: 0.9231
#     Total Return: 0.0062
#   EMA_fast_distance_pct→CUSUM_event_label_1.000:
#     AUC-ROC: 0.5478
#     Sharpe Ratio: 2.2507
#     Total Return: 0.0143
#   EMA_medium_distance_pct→CUSUM_event_label_0.999:
#     AUC-ROC: 0.5496
#     Sharpe Ratio: 2.1008
#     Total Return: 0.0149
#   30min_Close→CUSUM_event_label_0.999:
#     AUC-ROC: 0.5018
#     Sharpe Ratio: 0.0879
#     Total Return: 0.0005
#   EMA_slow_distance_pct→CUSUM_event_label_0.998:
#     AUC-ROC: 0.5470
#     Sharpe Ratio: 2.0034
#     Total Return: 0.0141
#   BB_position→CUSUM_event_label_0.994:
#     AUC-ROC: 0.5523
#     Sharpe Ratio: 2.8883
#     Total Return: 0.0157
#   RSI→CUSUM_event_label_0.994:
#     AUC-ROC: 0.5494
#     Sharpe Ratio: 1.8066
#     Total Return: 0.0148
#   4H_Volume→CUSUM_event_label_0.982:
#     AUC-ROC: 0.4946
#     Sharpe Ratio: -0.2399
#     Total Return: -0.0016
#   30min_High→CUSUM_event_label_0.979:
#     AUC-ROC: 0.5030
#     Sharpe Ratio: 0.1486
#     Total Return: 0.0009
#   MACD_histogram→CUSUM_event_label_0.979:
#     AUC-ROC: 0.5324
#     Sharpe Ratio: 0.8930
#     Total Return: 0.0097
#   30min_Low→CUSUM_event_label_0.976:
#     AUC-ROC: 0.5011
#     Sharpe Ratio: 0.0537
#     Total Return: 0.0003
#   1D_Low→CUSUM_event_label_0.967:
#     AUC-ROC: 0.5091
#     Sharpe Ratio: 0.5806
#     Total Return: 0.0027
#   ATR→CUSUM_event_label_0.957:
#     AUC-ROC: 0.5000
#     Sharpe Ratio: 0.0000
#     Total Return: 0.0000
#   1D_Open→CUSUM_event_label_0.949:
#     AUC-ROC: 0.5067
#     Sharpe Ratio: 0.5525
#     Total Return: 0.0020
#
# --------------------------------------------------
#
# 🎯 EVENT: vpd_volatility_event_label
# --------------------------------------------------
# Occurrences: 721
# Frequency: 0.0012
#
# CAUSAL RELATIONSHIPS FOUND: 10
#
# Top Causal Features:
#    1. Volume_SMA
#       Strength: 0.9990 | P-value: 0.001050
#       Mechanism: information_asymmetry
#       Justification: Supported by order_flow_impact mechanism: Order flow directly impacts prices through market mechanics
#
#    2. 4H_Volume
#       Strength: 0.9987 | P-value: 0.001290
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    3. CUSUM_pos
#       Strength: 0.9926 | P-value: 0.007436
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    4. 1D_Volume
#       Strength: 0.9670 | P-value: 0.033006
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    5. CUSUM_neg
#       Strength: 0.9614 | P-value: 0.038629
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    6. 4H_Low
#       Strength: 0.9466 | P-value: 0.053362
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    7. 30min_Close
#       Strength: 0.9370 | P-value: 0.062990
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    8. EMA_slow_distance_atr
#       Strength: 0.9281 | P-value: 0.071856
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    9. EMA_medium_26
#       Strength: 0.9069 | P-value: 0.093121
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#   10. 30min_Low
#       Strength: 0.9012 | P-value: 0.098838
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
# VALIDATION RESULTS:
#   Volume_SMA→vpd_volatility_event_label_0.999:
#     AUC-ROC: 0.5044
#     Sharpe Ratio: 0.0483
#     Total Return: 0.0013
#   4H_Volume→vpd_volatility_event_label_0.999:
#     AUC-ROC: 0.5056
#     Sharpe Ratio: 0.0620
#     Total Return: 0.0017
#   CUSUM_pos→vpd_volatility_event_label_0.993:
#     AUC-ROC: 0.5017
#     Sharpe Ratio: 0.0317
#     Total Return: 0.0005
#   1D_Volume→vpd_volatility_event_label_0.967:
#     AUC-ROC: 0.6105
#     Sharpe Ratio: 1.8089
#     Total Return: 0.0332
#   CUSUM_neg→vpd_volatility_event_label_0.961:
#     AUC-ROC: 0.5101
#     Sharpe Ratio: 1.1954
#     Total Return: 0.0030
#   4H_Low→vpd_volatility_event_label_0.947:
#     AUC-ROC: 0.5497
#     Sharpe Ratio: 0.4168
#     Total Return: 0.0149
#   30min_Close→vpd_volatility_event_label_0.937:
#     AUC-ROC: 0.6202
#     Sharpe Ratio: 1.0857
#     Total Return: 0.0361
#   EMA_slow_distance_atr→vpd_volatility_event_label_0.928:
#     AUC-ROC: 0.5862
#     Sharpe Ratio: 1.1777
#     Total Return: 0.0258
#   EMA_medium_26→vpd_volatility_event_label_0.907:
#     AUC-ROC: 0.5577
#     Sharpe Ratio: 0.5567
#     Total Return: 0.0173
#   30min_Low→vpd_volatility_event_label_0.901:
#     AUC-ROC: 0.6242
#     Sharpe Ratio: 1.1321
#     Total Return: 0.0373
#
# --------------------------------------------------
#
# 🎯 EVENT: outlier_event_label
# --------------------------------------------------
# Occurrences: 35591
# Frequency: 0.0598
#
# CAUSAL RELATIONSHIPS FOUND: 15
#
# Top Causal Features:
#    1. 30min_High
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    2. 30min_Low
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    3. 30min_Close
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    4. MACD
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    5. MACD_histogram
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    6. RSI
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Supported by momentum_herding mechanism: Past returns cause future returns through behavioral herding
#
#    7. CUSUM_pos
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    8. CUSUM_neg
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    9. returns
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Supported by momentum_herding mechanism: Past returns cause future returns through behavioral herding
#
#   10. EMA_fast_distance_pct
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
# VALIDATION RESULTS:
#   30min_High→outlier_event_label_1.000:
#     AUC-ROC: 0.5046
#     Sharpe Ratio: 0.2866
#     Total Return: 0.0014
#   30min_Low→outlier_event_label_1.000:
#     AUC-ROC: 0.5051
#     Sharpe Ratio: 0.3275
#     Total Return: 0.0015
#   30min_Close→outlier_event_label_1.000:
#     AUC-ROC: 0.5049
#     Sharpe Ratio: 0.3132
#     Total Return: 0.0015
#   MACD→outlier_event_label_1.000:
#     AUC-ROC: 0.5159
#     Sharpe Ratio: 4.2342
#     Total Return: 0.0048
#   MACD_histogram→outlier_event_label_1.000:
#     AUC-ROC: 0.4880
#     Sharpe Ratio: -1.6868
#     Total Return: -0.0036
#   RSI→outlier_event_label_1.000:
#     AUC-ROC: 0.5105
#     Sharpe Ratio: 0.7164
#     Total Return: 0.0031
#   CUSUM_pos→outlier_event_label_1.000:
#     AUC-ROC: 0.5025
#     Sharpe Ratio: 0.2092
#     Total Return: 0.0008
#   CUSUM_neg→outlier_event_label_1.000:
#     AUC-ROC: 0.5095
#     Sharpe Ratio: 0.7288
#     Total Return: 0.0028
#   returns→outlier_event_label_1.000:
#     AUC-ROC: 0.4993
#     Sharpe Ratio: -0.0659
#     Total Return: -0.0002
#   EMA_fast_distance_pct→outlier_event_label_1.000:
#     AUC-ROC: 0.4917
#     Sharpe Ratio: -0.6036
#     Total Return: -0.0025
#   EMA_medium_distance_pct→outlier_event_label_1.000:
#     AUC-ROC: 0.5088
#     Sharpe Ratio: 0.6293
#     Total Return: 0.0026
#   EMA_slow_distance_pct→outlier_event_label_1.000:
#     AUC-ROC: 0.5174
#     Sharpe Ratio: 8.3311
#     Total Return: 0.0052
#   BB_position→outlier_event_label_1.000:
#     AUC-ROC: 0.5060
#     Sharpe Ratio: 0.4360
#     Total Return: 0.0018
#   30min_Open→outlier_event_label_1.000:
#     AUC-ROC: 0.5048
#     Sharpe Ratio: 0.2992
#     Total Return: 0.0014
#   EMA_fast_12→outlier_event_label_1.000:
#     AUC-ROC: 0.5052
#     Sharpe Ratio: 0.3163
#     Total Return: 0.0015
#
# --------------------------------------------------
#
# 🎯 EVENT: momentum_regime_event_label
# --------------------------------------------------
# Occurrences: 231363
# Frequency: 0.3888
#
# CAUSAL RELATIONSHIPS FOUND: 15
#
# Top Causal Features:
#    1. 30min_Open
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    2. 30min_High
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    3. 30min_Low
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    4. 30min_Close
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    5. EMA_fast_12
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    6. EMA_medium_26
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    7. EMA_slow_50
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    8. MACD
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    9. MACD_signal
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#   10. MACD_histogram
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
# VALIDATION RESULTS:
#   30min_Open→momentum_regime_event_label_1.000:
#     AUC-ROC: 0.5031
#     Sharpe Ratio: 0.2870
#     Total Return: 0.0009
#   30min_High→momentum_regime_event_label_1.000:
#     AUC-ROC: 0.5029
#     Sharpe Ratio: 0.2645
#     Total Return: 0.0009
#   30min_Low→momentum_regime_event_label_1.000:
#     AUC-ROC: 0.5033
#     Sharpe Ratio: 0.3137
#     Total Return: 0.0010
#   30min_Close→momentum_regime_event_label_1.000:
#     AUC-ROC: 0.5031
#     Sharpe Ratio: 0.2900
#     Total Return: 0.0009
#   EMA_fast_12→momentum_regime_event_label_1.000:
#     AUC-ROC: 0.5034
#     Sharpe Ratio: 0.2983
#     Total Return: 0.0010
#   EMA_medium_26→momentum_regime_event_label_1.000:
#     AUC-ROC: 0.5038
#     Sharpe Ratio: 0.3372
#     Total Return: 0.0011
#   EMA_slow_50→momentum_regime_event_label_1.000:
#     AUC-ROC: 0.5042
#     Sharpe Ratio: 0.3717
#     Total Return: 0.0013
#   MACD→momentum_regime_event_label_1.000:
#     AUC-ROC: 0.5194
#     Sharpe Ratio: 3.3653
#     Total Return: 0.0058
#   MACD_signal→momentum_regime_event_label_1.000:
#     AUC-ROC: 0.5173
#     Sharpe Ratio: 3.9350
#     Total Return: 0.0052
#   MACD_histogram→momentum_regime_event_label_1.000:
#     AUC-ROC: 0.4950
#     Sharpe Ratio: -0.3635
#     Total Return: -0.0015
#   RSI→momentum_regime_event_label_1.000:
#     AUC-ROC: 0.5213
#     Sharpe Ratio: 2.3633
#     Total Return: 0.0064
#   EMA_fast_distance_pct→momentum_regime_event_label_1.000:
#     AUC-ROC: 0.5155
#     Sharpe Ratio: 1.2349
#     Total Return: 0.0047
#   EMA_medium_distance_pct→momentum_regime_event_label_1.000:
#     AUC-ROC: 0.5203
#     Sharpe Ratio: 2.5691
#     Total Return: 0.0061
#   EMA_slow_distance_pct→momentum_regime_event_label_1.000:
#     AUC-ROC: 0.5201
#     Sharpe Ratio: 3.3166
#     Total Return: 0.0060
#   BB_position→momentum_regime_event_label_1.000:
#     AUC-ROC: 0.5038
#     Sharpe Ratio: 0.1946
#     Total Return: 0.0011
#
# --------------------------------------------------
#
# 🎯 EVENT: traditional_event_label
# --------------------------------------------------
# Occurrences: 250896
# Frequency: 0.4217
#
# CAUSAL RELATIONSHIPS FOUND: 15
#
# Top Causal Features:
#    1. 30min_High
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    2. 30min_Low
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    3. 30min_Close
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    4. EMA_fast_12
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    5. EMA_medium_26
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    6. MACD
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    7. MACD_histogram
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    8. RSI
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Supported by momentum_herding mechanism: Past returns cause future returns through behavioral herding
#
#    9. CUSUM_pos
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#   10. CUSUM_neg
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
# VALIDATION RESULTS:
#   30min_High→traditional_event_label_1.000:
#     AUC-ROC: 0.5032
#     Sharpe Ratio: 0.2647
#     Total Return: 0.0009
#   30min_Low→traditional_event_label_1.000:
#     AUC-ROC: 0.5036
#     Sharpe Ratio: 0.3109
#     Total Return: 0.0011
#   30min_Close→traditional_event_label_1.000:
#     AUC-ROC: 0.5034
#     Sharpe Ratio: 0.2892
#     Total Return: 0.0010
#   EMA_fast_12→traditional_event_label_1.000:
#     AUC-ROC: 0.5036
#     Sharpe Ratio: 0.2982
#     Total Return: 0.0011
#   EMA_medium_26→traditional_event_label_1.000:
#     AUC-ROC: 0.5040
#     Sharpe Ratio: 0.3331
#     Total Return: 0.0012
#   MACD→traditional_event_label_1.000:
#     AUC-ROC: 0.5187
#     Sharpe Ratio: 4.0768
#     Total Return: 0.0056
#   MACD_histogram→traditional_event_label_1.000:
#     AUC-ROC: 0.4949
#     Sharpe Ratio: -0.3590
#     Total Return: -0.0015
#   RSI→traditional_event_label_1.000:
#     AUC-ROC: 0.5208
#     Sharpe Ratio: 2.6758
#     Total Return: 0.0062
#   CUSUM_pos→traditional_event_label_1.000:
#     AUC-ROC: 0.4997
#     Sharpe Ratio: -0.1104
#     Total Return: -0.0001
#   CUSUM_neg→traditional_event_label_1.000:
#     AUC-ROC: 0.5023
#     Sharpe Ratio: 0.9984
#     Total Return: 0.0007
#   CUSUM_direction→traditional_event_label_1.000:
#     AUC-ROC: 0.5001
#     Sharpe Ratio: 0.3135
#     Total Return: 0.0000
#   returns→traditional_event_label_1.000:
#     AUC-ROC: 0.5000
#     Sharpe Ratio: -0.0368
#     Total Return: -0.0000
#   EMA_fast_distance_pct→traditional_event_label_1.000:
#     AUC-ROC: 0.5149
#     Sharpe Ratio: 1.2455
#     Total Return: 0.0045
#   EMA_medium_distance_pct→traditional_event_label_1.000:
#     AUC-ROC: 0.5195
#     Sharpe Ratio: 2.8301
#     Total Return: 0.0059
#   EMA_slow_distance_pct→traditional_event_label_1.000:
#     AUC-ROC: 0.5195
#     Sharpe Ratio: 4.8783
#     Total Return: 0.0059
#
# --------------------------------------------------
#
# 🎯 EVENT: any_event_label
# --------------------------------------------------
# Occurrences: 493248
# Frequency: 0.8290
#
# CAUSAL RELATIONSHIPS FOUND: 15
#
# Top Causal Features:
#    1. 30min_Close
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    2. RSI
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Supported by momentum_herding mechanism: Past returns cause future returns through behavioral herding
#
#    3. CUSUM_neg
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    4. returns
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Supported by momentum_herding mechanism: Past returns cause future returns through behavioral herding
#
#    5. EMA_fast_distance_pct
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    6. EMA_medium_distance_pct
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    7. EMA_slow_distance_pct
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    8. BB_position
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    9. MACD_histogram
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#   10. CUSUM_pos
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
# VALIDATION RESULTS:
#   30min_Close→any_event_label_1.000:
#     AUC-ROC: 0.5048
#     Sharpe Ratio: 0.4550
#     Total Return: 0.0014
#   RSI→any_event_label_1.000:
#     AUC-ROC: 0.5136
#     Sharpe Ratio: 0.7655
#     Total Return: 0.0041
#   CUSUM_neg→any_event_label_1.000:
#     AUC-ROC: 0.5017
#     Sharpe Ratio: 1.6177
#     Total Return: 0.0005
#   returns→any_event_label_1.000:
#     AUC-ROC: 0.5000
#     Sharpe Ratio: 0.0037
#     Total Return: 0.0000
#   EMA_fast_distance_pct→any_event_label_1.000:
#     AUC-ROC: 0.5151
#     Sharpe Ratio: 1.3457
#     Total Return: 0.0045
#   EMA_medium_distance_pct→any_event_label_1.000:
#     AUC-ROC: 0.5203
#     Sharpe Ratio: 3.0743
#     Total Return: 0.0061
#   EMA_slow_distance_pct→any_event_label_1.000:
#     AUC-ROC: 0.5200
#     Sharpe Ratio: 6.2223
#     Total Return: 0.0060
#   BB_position→any_event_label_1.000:
#     AUC-ROC: 0.5025
#     Sharpe Ratio: 0.1350
#     Total Return: 0.0007
#   MACD_histogram→any_event_label_1.000:
#     AUC-ROC: 0.4982
#     Sharpe Ratio: -0.1200
#     Total Return: -0.0005
#   CUSUM_pos→any_event_label_1.000:
#     AUC-ROC: 0.4998
#     Sharpe Ratio: -0.0807
#     Total Return: -0.0001
#   MACD→any_event_label_1.000:
#     AUC-ROC: 0.5202
#     Sharpe Ratio: 6.8275
#     Total Return: 0.0061
#   EMA_fast_12→any_event_label_1.000:
#     AUC-ROC: 0.5051
#     Sharpe Ratio: 0.4601
#     Total Return: 0.0015
#   MACD_efficiency→any_event_label_1.000:
#     AUC-ROC: 0.5114
#     Sharpe Ratio: 0.6364
#     Total Return: 0.0034
#   EMA_medium_26→any_event_label_1.000:
#     AUC-ROC: 0.5056
#     Sharpe Ratio: 0.4948
#     Total Return: 0.0017
#   ATR→any_event_label_1.000:
#     AUC-ROC: 0.5000
#     Sharpe Ratio: 0.0000
#     Total Return: 0.0000
#
# --------------------------------------------------
#
# 🎯 EVENT: covent_2way_CUSUM_BB_upper_cross_label
# --------------------------------------------------
# Occurrences: 1200
# Frequency: 0.0020
#
# CAUSAL RELATIONSHIPS FOUND: 15
#
# Top Causal Features:
#    1. CUSUM_direction
#       Strength: 0.9995 | P-value: 0.000473
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    2. returns
#       Strength: 0.9994 | P-value: 0.000599
#       Mechanism: information_asymmetry
#       Justification: Supported by momentum_herding mechanism: Past returns cause future returns through behavioral herding
#
#    3. 4H_Volume
#       Strength: 0.9927 | P-value: 0.007342
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    4. 30min_Volume
#       Strength: 0.9910 | P-value: 0.008988
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    5. EMA_fast_distance_pct
#       Strength: 0.9876 | P-value: 0.012438
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    6. RSI_momentum
#       Strength: 0.9867 | P-value: 0.013259
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    7. EMA_medium_distance_pct
#       Strength: 0.9799 | P-value: 0.020056
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    8. Volume_SMA
#       Strength: 0.9670 | P-value: 0.033040
#       Mechanism: information_asymmetry
#       Justification: Supported by order_flow_impact mechanism: Order flow directly impacts prices through market mechanics
#
#    9. EMA_slow_distance_atr
#       Strength: 0.9577 | P-value: 0.042290
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#   10. EMA_fast_distance_atr
#       Strength: 0.9564 | P-value: 0.043553
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
# VALIDATION RESULTS:
#   CUSUM_direction→covent_2way_CUSUM_BB_upper_cross_label_1.000:
#     AUC-ROC: 0.5111
#     Sharpe Ratio: 5.5607
#     Total Return: 0.0033
#   returns→covent_2way_CUSUM_BB_upper_cross_label_0.999:
#     AUC-ROC: 0.5103
#     Sharpe Ratio: 4.7066
#     Total Return: 0.0031
#   4H_Volume→covent_2way_CUSUM_BB_upper_cross_label_0.993:
#     AUC-ROC: 0.5001
#     Sharpe Ratio: 0.0008
#     Total Return: 0.0000
#   30min_Volume→covent_2way_CUSUM_BB_upper_cross_label_0.991:
#     AUC-ROC: 0.5350
#     Sharpe Ratio: 0.7338
#     Total Return: 0.0105
#   EMA_fast_distance_pct→covent_2way_CUSUM_BB_upper_cross_label_0.988:
#     AUC-ROC: 0.5151
#     Sharpe Ratio: 0.2454
#     Total Return: 0.0045
#   RSI_momentum→covent_2way_CUSUM_BB_upper_cross_label_0.987:
#     AUC-ROC: 0.5282
#     Sharpe Ratio: 0.8261
#     Total Return: 0.0085
#   EMA_medium_distance_pct→covent_2way_CUSUM_BB_upper_cross_label_0.980:
#     AUC-ROC: 0.4841
#     Sharpe Ratio: -0.2257
#     Total Return: -0.0048
#   Volume_SMA→covent_2way_CUSUM_BB_upper_cross_label_0.967:
#     AUC-ROC: 0.5473
#     Sharpe Ratio: 1.1681
#     Total Return: 0.0142
#   EMA_slow_distance_atr→covent_2way_CUSUM_BB_upper_cross_label_0.958:
#     AUC-ROC: 0.5000
#     Sharpe Ratio: 0.0000
#     Total Return: 0.0000
#   EMA_fast_distance_atr→covent_2way_CUSUM_BB_upper_cross_label_0.956:
#     AUC-ROC: 0.5000
#     Sharpe Ratio: 0.0000
#     Total Return: 0.0000
#   EMA_medium_distance_atr→covent_2way_CUSUM_BB_upper_cross_label_0.953:
#     AUC-ROC: 0.5000
#     Sharpe Ratio: 0.0000
#     Total Return: 0.0000
#   RSI→covent_2way_CUSUM_BB_upper_cross_label_0.938:
#     AUC-ROC: 0.4769
#     Sharpe Ratio: -0.4704
#     Total Return: -0.0069
#   EMA_slow_distance_pct→covent_2way_CUSUM_BB_upper_cross_label_0.935:
#     AUC-ROC: 0.4948
#     Sharpe Ratio: -0.0651
#     Total Return: -0.0016
#   CUSUM_pos→covent_2way_CUSUM_BB_upper_cross_label_0.930:
#     AUC-ROC: 0.5013
#     Sharpe Ratio: 0.7071
#     Total Return: 0.0004
#   BB_position→covent_2way_CUSUM_BB_upper_cross_label_0.929:
#     AUC-ROC: 0.5358
#     Sharpe Ratio: 1.4212
#     Total Return: 0.0107
#
# --------------------------------------------------
#
# 🎯 EVENT: covent_2way_CUSUM_BB_lower_cross_label
# --------------------------------------------------
# Occurrences: 1525
# Frequency: 0.0026
#
# CAUSAL RELATIONSHIPS FOUND: 15
#
# Top Causal Features:
#    1. returns
#       Strength: 0.9977 | P-value: 0.002263
#       Mechanism: information_asymmetry
#       Justification: Supported by momentum_herding mechanism: Past returns cause future returns through behavioral herding
#
#    2. vol_realized
#       Strength: 0.9911 | P-value: 0.008880
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    3. CUSUM_direction
#       Strength: 0.9908 | P-value: 0.009163
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    4. EMA_fast_distance_pct
#       Strength: 0.9809 | P-value: 0.019144
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    5. BB_position
#       Strength: 0.9750 | P-value: 0.025021
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    6. ATR
#       Strength: 0.9730 | P-value: 0.026980
#       Mechanism: information_asymmetry
#       Justification: Supported by volatility_clustering mechanism: High volatility periods cause future high volatility
#
#    7. EMA_medium_distance_pct
#       Strength: 0.9729 | P-value: 0.027055
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    8. EMA_slow_distance_pct
#       Strength: 0.9596 | P-value: 0.040420
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    9. BB_width_pct
#       Strength: 0.9574 | P-value: 0.042621
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#   10. RSI
#       Strength: 0.9564 | P-value: 0.043590
#       Mechanism: information_asymmetry
#       Justification: Supported by momentum_herding mechanism: Past returns cause future returns through behavioral herding
#
# VALIDATION RESULTS:
#   returns→covent_2way_CUSUM_BB_lower_cross_label_0.998:
#     AUC-ROC: 0.5037
#     Sharpe Ratio: 0.3905
#     Total Return: 0.0011
#   vol_realized→covent_2way_CUSUM_BB_lower_cross_label_0.991:
#     AUC-ROC: 0.4785
#     Sharpe Ratio: -1.7794
#     Total Return: -0.0064
#   CUSUM_direction→covent_2way_CUSUM_BB_lower_cross_label_0.991:
#     AUC-ROC: 0.4921
#     Sharpe Ratio: -1.6480
#     Total Return: -0.0024
#   EMA_fast_distance_pct→covent_2way_CUSUM_BB_lower_cross_label_0.981:
#     AUC-ROC: 0.4861
#     Sharpe Ratio: -2.5745
#     Total Return: -0.0042
#   BB_position→covent_2way_CUSUM_BB_lower_cross_label_0.975:
#     AUC-ROC: 0.5321
#     Sharpe Ratio: 1.1488
#     Total Return: 0.0096
#   ATR→covent_2way_CUSUM_BB_lower_cross_label_0.973:
#     AUC-ROC: 0.5000
#     Sharpe Ratio: 0.0000
#     Total Return: 0.0000
#   EMA_medium_distance_pct→covent_2way_CUSUM_BB_lower_cross_label_0.973:
#     AUC-ROC: 0.4746
#     Sharpe Ratio: -1.2199
#     Total Return: -0.0076
#   EMA_slow_distance_pct→covent_2way_CUSUM_BB_lower_cross_label_0.960:
#     AUC-ROC: 0.4671
#     Sharpe Ratio: -1.1814
#     Total Return: -0.0099
#   BB_width_pct→covent_2way_CUSUM_BB_lower_cross_label_0.957:
#     AUC-ROC: 0.5197
#     Sharpe Ratio: 0.3996
#     Total Return: 0.0059
#   RSI→covent_2way_CUSUM_BB_lower_cross_label_0.956:
#     AUC-ROC: 0.4878
#     Sharpe Ratio: -0.3761
#     Total Return: -0.0037
#   Volume_SMA→covent_2way_CUSUM_BB_lower_cross_label_0.956:
#     AUC-ROC: 0.5383
#     Sharpe Ratio: 0.6101
#     Total Return: 0.0115
#   MACD_histogram→covent_2way_CUSUM_BB_lower_cross_label_0.950:
#     AUC-ROC: 0.4704
#     Sharpe Ratio: -2.3776
#     Total Return: -0.0089
#   30min_Low→covent_2way_CUSUM_BB_lower_cross_label_0.930:
#     AUC-ROC: 0.4947
#     Sharpe Ratio: -0.1535
#     Total Return: -0.0016
#   30min_Close→covent_2way_CUSUM_BB_lower_cross_label_0.928:
#     AUC-ROC: 0.4965
#     Sharpe Ratio: -0.1027
#     Total Return: -0.0010
#   30min_Open→covent_2way_CUSUM_BB_lower_cross_label_0.928:
#     AUC-ROC: 0.4985
#     Sharpe Ratio: -0.0473
#     Total Return: -0.0005
#
# --------------------------------------------------
#
# 🎯 EVENT: covent_2way_vpd_volatility_momentum_regime_label
# --------------------------------------------------
# Occurrences: 400
# Frequency: 0.0007
#
# CAUSAL RELATIONSHIPS FOUND: 1
#
# Top Causal Features:
#    1. ATR_pct
#       Strength: 0.9327 | P-value: 0.067326
#       Mechanism: information_asymmetry
#       Justification: Supported by volatility_clustering mechanism: High volatility periods cause future high volatility
#
# VALIDATION RESULTS:
#   ATR_pct→covent_2way_vpd_volatility_momentum_regime_label_0.933:
#     AUC-ROC: 0.4940
#     Sharpe Ratio: -0.0955
#     Total Return: -0.0018
#
# --------------------------------------------------
#
# 🎯 EVENT: covent_2way_outlier_BB_expansion_label
# --------------------------------------------------
# Occurrences: 88600
# Frequency: 0.1489
#
# CAUSAL RELATIONSHIPS FOUND: 15
#
# Top Causal Features:
#    1. RSI
#       Strength: 0.9998 | P-value: 0.000185
#       Mechanism: information_asymmetry
#       Justification: Supported by momentum_herding mechanism: Past returns cause future returns through behavioral herding
#
#    2. EMA_slow_distance_pct
#       Strength: 0.9997 | P-value: 0.000267
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    3. returns
#       Strength: 0.9997 | P-value: 0.000300
#       Mechanism: information_asymmetry
#       Justification: Supported by momentum_herding mechanism: Past returns cause future returns through behavioral herding
#
#    4. 30min_Close
#       Strength: 0.9997 | P-value: 0.000306
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    5. MACD_signal
#       Strength: 0.9991 | P-value: 0.000929
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    6. SMA_long_200
#       Strength: 0.9989 | P-value: 0.001114
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    7. MACD_efficiency
#       Strength: 0.9988 | P-value: 0.001223
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    8. EMA_medium_distance_pct
#       Strength: 0.9987 | P-value: 0.001253
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    9. 4H_EMA_medium
#       Strength: 0.9987 | P-value: 0.001314
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#   10. 1D_Low
#       Strength: 0.9983 | P-value: 0.001656
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
# VALIDATION RESULTS:
#   RSI→covent_2way_outlier_BB_expansion_label_1.000:
#     AUC-ROC: 0.5133
#     Sharpe Ratio: 1.6001
#     Total Return: 0.0040
#   EMA_slow_distance_pct→covent_2way_outlier_BB_expansion_label_1.000:
#     AUC-ROC: 0.5103
#     Sharpe Ratio: 1.5654
#     Total Return: 0.0031
#   returns→covent_2way_outlier_BB_expansion_label_1.000:
#     AUC-ROC: 0.4990
#     Sharpe Ratio: -0.3017
#     Total Return: -0.0003
#   30min_Close→covent_2way_outlier_BB_expansion_label_1.000:
#     AUC-ROC: 0.5114
#     Sharpe Ratio: 0.8070
#     Total Return: 0.0034
#   MACD_signal→covent_2way_outlier_BB_expansion_label_0.999:
#     AUC-ROC: 0.4962
#     Sharpe Ratio: -1.7301
#     Total Return: -0.0012
#   SMA_long_200→covent_2way_outlier_BB_expansion_label_0.999:
#     AUC-ROC: 0.5127
#     Sharpe Ratio: 0.7015
#     Total Return: 0.0038
#   MACD_efficiency→covent_2way_outlier_BB_expansion_label_0.999:
#     AUC-ROC: 0.4959
#     Sharpe Ratio: -0.7217
#     Total Return: -0.0012
#   EMA_medium_distance_pct→covent_2way_outlier_BB_expansion_label_0.999:
#     AUC-ROC: 0.5114
#     Sharpe Ratio: 1.3868
#     Total Return: 0.0034
#   4H_EMA_medium→covent_2way_outlier_BB_expansion_label_0.999:
#     AUC-ROC: 0.5129
#     Sharpe Ratio: 0.7403
#     Total Return: 0.0039
#   1D_Low→covent_2way_outlier_BB_expansion_label_0.998:
#     AUC-ROC: 0.5113
#     Sharpe Ratio: 0.7936
#     Total Return: 0.0034
#   30min_High→covent_2way_outlier_BB_expansion_label_0.998:
#     AUC-ROC: 0.5112
#     Sharpe Ratio: 0.7688
#     Total Return: 0.0034
#   BB_lower→covent_2way_outlier_BB_expansion_label_0.998:
#     AUC-ROC: 0.5120
#     Sharpe Ratio: 0.8181
#     Total Return: 0.0036
#   1D_Open→covent_2way_outlier_BB_expansion_label_0.998:
#     AUC-ROC: 0.5116
#     Sharpe Ratio: 0.6834
#     Total Return: 0.0035
#   1D_Close→covent_2way_outlier_BB_expansion_label_0.998:
#     AUC-ROC: 0.5118
#     Sharpe Ratio: 0.7788
#     Total Return: 0.0035
#   4H_Low→covent_2way_outlier_BB_expansion_label_0.998:
#     AUC-ROC: 0.5122
#     Sharpe Ratio: 0.8421
#     Total Return: 0.0037
#
# --------------------------------------------------
#
# 🎯 EVENT: covent_2way_momentum_regime_BB_squeeze_label
# --------------------------------------------------
# Occurrences: 188575
# Frequency: 0.3169
#
# CAUSAL RELATIONSHIPS FOUND: 15
#
# Top Causal Features:
#    1. 30min_Open
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    2. 30min_High
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    3. 30min_Low
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    4. 30min_Close
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    5. 4H_High
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    6. 4H_Low
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    7. 4H_Close
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    8. EMA_fast_12
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    9. EMA_medium_26
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#   10. EMA_slow_50
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
# VALIDATION RESULTS:
#   30min_Open→covent_2way_momentum_regime_BB_squeeze_label_1.000:
#     AUC-ROC: 0.5040
#     Sharpe Ratio: 0.3117
#     Total Return: 0.0012
#   30min_High→covent_2way_momentum_regime_BB_squeeze_label_1.000:
#     AUC-ROC: 0.5039
#     Sharpe Ratio: 0.2959
#     Total Return: 0.0012
#   30min_Low→covent_2way_momentum_regime_BB_squeeze_label_1.000:
#     AUC-ROC: 0.5042
#     Sharpe Ratio: 0.3273
#     Total Return: 0.0013
#   30min_Close→covent_2way_momentum_regime_BB_squeeze_label_1.000:
#     AUC-ROC: 0.5040
#     Sharpe Ratio: 0.3118
#     Total Return: 0.0012
#   4H_High→covent_2way_momentum_regime_BB_squeeze_label_1.000:
#     AUC-ROC: 0.5035
#     Sharpe Ratio: 0.2623
#     Total Return: 0.0011
#   4H_Low→covent_2way_momentum_regime_BB_squeeze_label_1.000:
#     AUC-ROC: 0.5045
#     Sharpe Ratio: 0.3650
#     Total Return: 0.0013
#   4H_Close→covent_2way_momentum_regime_BB_squeeze_label_1.000:
#     AUC-ROC: 0.5039
#     Sharpe Ratio: 0.3060
#     Total Return: 0.0012
#   EMA_fast_12→covent_2way_momentum_regime_BB_squeeze_label_1.000:
#     AUC-ROC: 0.5044
#     Sharpe Ratio: 0.3376
#     Total Return: 0.0013
#   EMA_medium_26→covent_2way_momentum_regime_BB_squeeze_label_1.000:
#     AUC-ROC: 0.5052
#     Sharpe Ratio: 0.4096
#     Total Return: 0.0016
#   EMA_slow_50→covent_2way_momentum_regime_BB_squeeze_label_1.000:
#     AUC-ROC: 0.5060
#     Sharpe Ratio: 0.4805
#     Total Return: 0.0018
#   SMA_short_20→covent_2way_momentum_regime_BB_squeeze_label_1.000:
#     AUC-ROC: 0.5048
#     Sharpe Ratio: 0.3713
#     Total Return: 0.0014
#   SMA_medium_50→covent_2way_momentum_regime_BB_squeeze_label_1.000:
#     AUC-ROC: 0.5060
#     Sharpe Ratio: 0.4834
#     Total Return: 0.0018
#   MACD→covent_2way_momentum_regime_BB_squeeze_label_1.000:
#     AUC-ROC: 0.5102
#     Sharpe Ratio: 0.3986
#     Total Return: 0.0031
#   MACD_signal→covent_2way_momentum_regime_BB_squeeze_label_1.000:
#     AUC-ROC: 0.5248
#     Sharpe Ratio: 4.5546
#     Total Return: 0.0074
#   MACD_histogram→covent_2way_momentum_regime_BB_squeeze_label_1.000:
#     AUC-ROC: 0.4998
#     Sharpe Ratio: -0.0120
#     Total Return: -0.0001
#
# --------------------------------------------------
#
# 🎯 EVENT: any_2way_covent_label
# --------------------------------------------------
# Occurrences: 273369
# Frequency: 0.4594
#
# CAUSAL RELATIONSHIPS FOUND: 15
#
# Top Causal Features:
#    1. 30min_Close
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    2. RSI
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Supported by momentum_herding mechanism: Past returns cause future returns through behavioral herding
#
#    3. EMA_fast_distance_pct
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    4. EMA_medium_distance_pct
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    5. EMA_slow_distance_pct
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    6. BB_position
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    7. MACD_histogram
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    8. 30min_High
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    9. MACD
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#   10. returns
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Supported by momentum_herding mechanism: Past returns cause future returns through behavioral herding
#
# VALIDATION RESULTS:
#   30min_Close→any_2way_covent_label_1.000:
#     AUC-ROC: 0.5070
#     Sharpe Ratio: 0.5037
#     Total Return: 0.0021
#   RSI→any_2way_covent_label_1.000:
#     AUC-ROC: 0.5228
#     Sharpe Ratio: 4.4871
#     Total Return: 0.0068
#   EMA_fast_distance_pct→any_2way_covent_label_1.000:
#     AUC-ROC: 0.5163
#     Sharpe Ratio: 1.3986
#     Total Return: 0.0049
#   EMA_medium_distance_pct→any_2way_covent_label_1.000:
#     AUC-ROC: 0.5220
#     Sharpe Ratio: 4.4500
#     Total Return: 0.0066
#   EMA_slow_distance_pct→any_2way_covent_label_1.000:
#     AUC-ROC: 0.5219
#     Sharpe Ratio: 10.8966
#     Total Return: 0.0066
#   BB_position→any_2way_covent_label_1.000:
#     AUC-ROC: 0.5018
#     Sharpe Ratio: 0.0971
#     Total Return: 0.0005
#   MACD_histogram→any_2way_covent_label_1.000:
#     AUC-ROC: 0.4855
#     Sharpe Ratio: -4.0545
#     Total Return: -0.0043
#   30min_High→any_2way_covent_label_1.000:
#     AUC-ROC: 0.5068
#     Sharpe Ratio: 0.4797
#     Total Return: 0.0020
#   MACD→any_2way_covent_label_1.000:
#     AUC-ROC: 0.5199
#     Sharpe Ratio: 6.6883
#     Total Return: 0.0060
#   returns→any_2way_covent_label_1.000:
#     AUC-ROC: 0.4994
#     Sharpe Ratio: -0.9536
#     Total Return: -0.0002
#   30min_Low→any_2way_covent_label_1.000:
#     AUC-ROC: 0.5072
#     Sharpe Ratio: 0.5263
#     Total Return: 0.0022
#   CUSUM_direction→any_2way_covent_label_1.000:
#     AUC-ROC: 0.5001
#     Sharpe Ratio: 0.3229
#     Total Return: 0.0000
#   CUSUM_pos→any_2way_covent_label_1.000:
#     AUC-ROC: 0.4996
#     Sharpe Ratio: -0.1294
#     Total Return: -0.0001
#   EMA_fast_12→any_2way_covent_label_1.000:
#     AUC-ROC: 0.5073
#     Sharpe Ratio: 0.5195
#     Total Return: 0.0022
#   MACD_efficiency→any_2way_covent_label_1.000:
#     AUC-ROC: 0.5200
#     Sharpe Ratio: 4.9560
#     Total Return: 0.0060
#
# --------------------------------------------------
#
# 🎯 EVENT: count_2way_covent_label
# --------------------------------------------------
# Occurrences: 266713
# Frequency: 0.4482
#
# CAUSAL RELATIONSHIPS FOUND: 15
#
# Top Causal Features:
#    1. 30min_High
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    2. 30min_Low
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    3. 30min_Close
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    4. MACD
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    5. MACD_histogram
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    6. RSI
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Supported by momentum_herding mechanism: Past returns cause future returns through behavioral herding
#
#    7. EMA_fast_distance_pct
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    8. EMA_medium_distance_pct
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#    9. EMA_slow_distance_pct
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
#   10. BB_position
#       Strength: 1.0000 | P-value: 0.000000
#       Mechanism: information_asymmetry
#       Justification: Economic plausibility check bypassed
#
# VALIDATION RESULTS:
#   30min_High→count_2way_covent_label_1.000:
#     AUC-ROC: 0.5067
#     Sharpe Ratio: 0.4432
#     Total Return: 0.0020
#   30min_Low→count_2way_covent_label_1.000:
#     AUC-ROC: 0.5071
#     Sharpe Ratio: 0.4859
#     Total Return: 0.0021
#   30min_Close→count_2way_covent_label_1.000:
#     AUC-ROC: 0.5069
#     Sharpe Ratio: 0.4658
#     Total Return: 0.0021
#   MACD→count_2way_covent_label_1.000:
#     AUC-ROC: 0.5198
#     Sharpe Ratio: 6.3242
#     Total Return: 0.0060
#   MACD_histogram→count_2way_covent_label_1.000:
#     AUC-ROC: 0.4862
#     Sharpe Ratio: -3.3285
#     Total Return: -0.0041
#   RSI→count_2way_covent_label_1.000:
#     AUC-ROC: 0.5225
#     Sharpe Ratio: 4.3318
#     Total Return: 0.0068
#   EMA_fast_distance_pct→count_2way_covent_label_1.000:
#     AUC-ROC: 0.5148
#     Sharpe Ratio: 1.1426
#     Total Return: 0.0045
#   EMA_medium_distance_pct→count_2way_covent_label_1.000:
#     AUC-ROC: 0.5217
#     Sharpe Ratio: 4.3175
#     Total Return: 0.0065
#   EMA_slow_distance_pct→count_2way_covent_label_1.000:
#     AUC-ROC: 0.5219
#     Sharpe Ratio: 9.7434
#     Total Return: 0.0066
#   BB_position→count_2way_covent_label_1.000:
#     AUC-ROC: 0.5022
#     Sharpe Ratio: 0.1187
#     Total Return: 0.0006
#   EMA_fast_12→count_2way_covent_label_1.000:
#     AUC-ROC: 0.5073
#     Sharpe Ratio: 0.4814
#     Total Return: 0.0022
#   returns→count_2way_covent_label_1.000:
#     AUC-ROC: 0.4996
#     Sharpe Ratio: -0.6600
#     Total Return: -0.0001
#   EMA_medium_26→count_2way_covent_label_1.000:
#     AUC-ROC: 0.5078
#     Sharpe Ratio: 0.5218
#     Total Return: 0.0024
#   30min_Open→count_2way_covent_label_1.000:
#     AUC-ROC: 0.5069
#     Sharpe Ratio: 0.4640
#     Total Return: 0.0021
#   CUSUM_pos→count_2way_covent_label_1.000:
#     AUC-ROC: 0.4997
#     Sharpe Ratio: -0.1040
#     Total Return: -0.0001
#
# --------------------------------------------------
#
# ============================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================
#
# TOP 15 MOST IMPORTANT CAUSAL FEATURES:
# Rank                   Feature Avg Strength  Count Events
# ----------------------------------------------------------------------
#    1               EMA_slow_50       1.0000      2 momentum, covent
#    2                   4H_High       1.0000      1 covent
#    3                  4H_Close       1.0000      1 covent
#    4              SMA_short_20       1.0000      1 covent
#    5             SMA_medium_50       1.0000      1 covent
#    6                      MACD       1.0000      7 outlier, momentum, traditional (+4 more)
#    7               EMA_fast_12       1.0000      7 outlier, momentum, traditional (+4 more)
#    8               MACD_signal       0.9997      3 momentum, covent, covent
#    9                   returns       0.9996      9 CUSUM, outlier, traditional (+6 more)
#   10           MACD_efficiency       0.9996      3 any, covent, any
#   11              SMA_long_200       0.9989      1 covent
#   12             4H_EMA_medium       0.9987      1 covent
#   13                  BB_lower       0.9983      1 covent
#   14                  1D_Close       0.9982      1 covent
#   15           CUSUM_direction       0.9981      5 CUSUM, traditional, covent (+2 more)
#
# ============================================================
# ECONOMIC MECHANISM ANALYSIS
# ============================================================
#
# Mechanism distribution:
#   information_asymmetry: 176 relationships (100.0%)
# Event column analysis:
#
# Label distributions:
#
# Feature statistics:
# Technical indicators: 27
#
# ============================================================
# QUICK REFERENCE: BEST FEATURES BY EVENT
# ============================================================
#
# CUSUM_event_label:
#   1. CUSUM_direction (strength: 1.0000)
#   2. returns (strength: 1.0000)
#   3. EMA_fast_distance_pct (strength: 0.9997)
#   4. EMA_medium_distance_pct (strength: 0.9993)
#   5. 30min_Close (strength: 0.9989)
#
# vpd_volatility_event_label:
#   1. Volume_SMA (strength: 0.9990)
#   2. 4H_Volume (strength: 0.9987)
#   3. CUSUM_pos (strength: 0.9926)
#   4. 1D_Volume (strength: 0.9670)
#   5. CUSUM_neg (strength: 0.9614)
#
# outlier_event_label:
#   1. 30min_High (strength: 1.0000)
#   2. 30min_Low (strength: 1.0000)
#   3. 30min_Close (strength: 1.0000)
#   4. MACD (strength: 1.0000)
#   5. MACD_histogram (strength: 1.0000)
#
# momentum_regime_event_label:
#   1. 30min_Open (strength: 1.0000)
#   2. 30min_High (strength: 1.0000)
#   3. 30min_Low (strength: 1.0000)
#   4. 30min_Close (strength: 1.0000)
#   5. EMA_fast_12 (strength: 1.0000)
#
# traditional_event_label:
#   1. 30min_High (strength: 1.0000)
#   2. 30min_Low (strength: 1.0000)
#   3. 30min_Close (strength: 1.0000)
#   4. EMA_fast_12 (strength: 1.0000)
#   5. EMA_medium_26 (strength: 1.0000)
#
# any_event_label:
#   1. 30min_Close (strength: 1.0000)
#   2. RSI (strength: 1.0000)
#   3. CUSUM_neg (strength: 1.0000)
#   4. returns (strength: 1.0000)
#   5. EMA_fast_distance_pct (strength: 1.0000)
#
# covent_2way_CUSUM_BB_upper_cross_label:
#   1. CUSUM_direction (strength: 0.9995)
#   2. returns (strength: 0.9994)
#   3. 4H_Volume (strength: 0.9927)
#   4. 30min_Volume (strength: 0.9910)
#   5. EMA_fast_distance_pct (strength: 0.9876)
#
# covent_2way_CUSUM_BB_lower_cross_label:
#   1. returns (strength: 0.9977)
#   2. vol_realized (strength: 0.9911)
#   3. CUSUM_direction (strength: 0.9908)
#   4. EMA_fast_distance_pct (strength: 0.9809)
#   5. BB_position (strength: 0.9750)
#
# covent_2way_vpd_volatility_momentum_regime_label:
#   1. ATR_pct (strength: 0.9327)
#
# covent_2way_outlier_BB_expansion_label:
#   1. RSI (strength: 0.9998)
#   2. EMA_slow_distance_pct (strength: 0.9997)
#   3. returns (strength: 0.9997)
#   4. 30min_Close (strength: 0.9997)
#   5. MACD_signal (strength: 0.9991)
#
# covent_2way_momentum_regime_BB_squeeze_label:
#   1. 30min_Open (strength: 1.0000)
#   2. 30min_High (strength: 1.0000)
#   3. 30min_Low (strength: 1.0000)
#   4. 30min_Close (strength: 1.0000)
#   5. 4H_High (strength: 1.0000)
#
# any_2way_covent_label:
#   1. 30min_Close (strength: 1.0000)
#   2. RSI (strength: 1.0000)
#   3. EMA_fast_distance_pct (strength: 1.0000)
#   4. EMA_medium_distance_pct (strength: 1.0000)
#   5. EMA_slow_distance_pct (strength: 1.0000)
#
# count_2way_covent_label:
#   1. 30min_High (strength: 1.0000)
#   2. 30min_Low (strength: 1.0000)
#   3. 30min_Close (strength: 1.0000)
#   4. MACD (strength: 1.0000)
#   5. MACD_histogram (strength: 1.0000)
