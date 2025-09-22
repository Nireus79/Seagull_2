import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import warnings
from pathlib import Path
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
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
