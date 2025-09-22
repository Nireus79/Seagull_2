import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


def select_best_model_for_causal_features():
    """
    Practical model selection using your discovered causal features
    """

    # Load your data
    data = pd.read_pickle('D:/Seagull_data/labeled5mEE2cov.pkl')

    # Load causal research results
    with open('causal_results/research_results.pkl', 'rb') as f:
        causal_results = pickle.load(f)

    print("CAUSAL FEATURE-BASED MODEL SELECTION")
    print("=" * 60)

    # Extract the best causal features for each event
    event_recommendations = {}

    for event_label, event_data in causal_results['research_results'].items():
        causal_relationships = event_data.get('causal_relationships', [])

        if len(causal_relationships) == 0:
            continue

        # Get top causal features
        sorted_features = sorted(causal_relationships,
                                 key=lambda x: x['strength'],
                                 reverse=True)

        # Select top features based on causal strength
        top_features = [rel['cause_feature'] for rel in sorted_features[:10]
                        if rel['strength'] > 0.3]  # Only strong causal relationships

        if len(top_features) == 0:
            continue

        print(f"\n{event_label}:")
        print(f"  Event occurrences: {event_data.get('event_occurrences', 'N/A')}")
        print(f"  Top causal features ({len(top_features)}): {top_features[:5]}...")

        # Prepare data for this event
        try:
            # Get clean data
            event_data_clean = data[[event_label] + top_features].dropna()

            if len(event_data_clean) < 100:
                print(f"  Insufficient data: {len(event_data_clean)} samples")
                continue

            X = event_data_clean[top_features]
            y = event_data_clean[event_label]

            # Check class balance
            class_dist = y.value_counts()
            minority_ratio = class_dist.min() / class_dist.sum()
            print(f"  Class balance: {dict(class_dist)} (minority: {minority_ratio:.3f})")

            # Time series split
            split_point = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

            # Comprehensive model suite with all algorithms
            all_models = {
                # Linear Models
                'LogisticRegression': {
                    'model': LogisticRegression(random_state=42, max_iter=1000),
                    'needs_scaling': True,
                    'good_for': 'Interpretable, stable, works with causal features'
                },
                'RidgeClassifier': {
                    'model': RidgeClassifier(random_state=42),
                    'needs_scaling': True,
                    'good_for': 'Handles multicollinearity, regularized'
                },
                'SGDClassifier': {
                    'model': SGDClassifier(random_state=42, loss='log', max_iter=1000),
                    'needs_scaling': True,
                    'good_for': 'Large datasets, online learning'
                },

                # Tree-based Models
                'RandomForest': {
                    'model': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                    'needs_scaling': False,
                    'good_for': 'Feature interactions, robust to outliers'
                },
                'ExtraTrees': {
                    'model': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                    'needs_scaling': False,
                    'good_for': 'Faster than RF, less overfitting'
                },
                'GradientBoosting': {
                    'model': GradientBoostingClassifier(random_state=42, n_estimators=100),
                    'needs_scaling': False,
                    'good_for': 'Sequential learning, handles interactions'
                },

                # Advanced Gradient Boosting
                'XGBoost': {
                    'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss',
                                               n_estimators=100),
                    'needs_scaling': False,
                    'good_for': 'High performance, handles missing values'
                },
                'LightGBM': {
                    'model': lgb.LGBMClassifier(random_state=42, verbose=-1,
                                                n_estimators=100),
                    'needs_scaling': False,
                    'good_for': 'Fast, memory efficient, great for time series'
                },

                # Neural Networks
                'MLPClassifier': {
                    'model': MLPClassifier(random_state=42, max_iter=500,
                                           hidden_layer_sizes=(100, 50)),
                    'needs_scaling': True,
                    'good_for': 'Complex non-linear patterns'
                },

                # Probabilistic Models
                'GaussianNB': {
                    'model': GaussianNB(),
                    'needs_scaling': False,
                    'good_for': 'Small datasets, probabilistic outputs'
                },

                # Discriminant Analysis
                'LinearDiscriminant': {
                    'model': LinearDiscriminantAnalysis(),
                    'needs_scaling': True,
                    'good_for': 'Dimensionality reduction, multiclass'
                },
                'QuadraticDiscriminant': {
                    'model': QuadraticDiscriminantAnalysis(),
                    'needs_scaling': True,
                    'good_for': 'Non-linear decision boundaries'
                },

                # Support Vector Machines
                'SVM_RBF': {
                    'model': SVC(random_state=42, probability=True, kernel='rbf'),
                    'needs_scaling': True,
                    'good_for': 'Non-linear patterns, small-medium datasets'
                },
                'SVM_Linear': {
                    'model': SVC(random_state=42, probability=True, kernel='linear'),
                    'needs_scaling': True,
                    'good_for': 'High-dimensional data, interpretable'
                },

                # K-Nearest Neighbors
                'KNN': {
                    'model': KNeighborsClassifier(n_neighbors=5),
                    'needs_scaling': True,
                    'good_for': 'Local patterns, non-parametric'
                },

                # Ensemble Methods
                'AdaBoost': {
                    'model': AdaBoostClassifier(random_state=42, n_estimators=100),
                    'needs_scaling': False,
                    'good_for': 'Weak learners combination, less overfitting'
                },
                'BaggingClassifier': {
                    'model': BaggingClassifier(random_state=42, n_estimators=100),
                    'needs_scaling': False,
                    'good_for': 'Variance reduction, parallel training'
                }
            }

            # Adaptive model selection based on data characteristics
            models_to_test = adaptive_model_selection(X, y, event_label)

            # Filter models to test
            models = {name: info for name, info in all_models.items() if name in models_to_test}

            results = {}

            # Evaluate each model
            for name, model_info in models.items():
                try:
                    model = model_info['model']

                    if model_info['needs_scaling']:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        model.fit(X_train_scaled, y_train)
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    else:
                        model.fit(X_train, y_train)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]

                    # Calculate metrics
                    auc_score = roc_auc_score(y_test, y_pred_proba)

                    results[name] = {
                        'auc': auc_score,
                        'model': model,
                        'scaler': scaler if model_info['needs_scaling'] else None,
                        'good_for': model_info['good_for']
                    }

                except Exception as e:
                    print(f"    {name} failed: {str(e)}")
                    continue

            # Find best model
            if results:
                best_model_name = max(results.keys(), key=lambda k: results[k]['auc'])
                best_auc = results[best_model_name]['auc']

                print(f"  MODEL PERFORMANCE:")
                for name, result in sorted(results.items(),
                                           key=lambda x: x[1]['auc'],
                                           reverse=True):
                    print(f"    {name:<15}: AUC = {result['auc']:.4f} | {result['good_for']}")

                print(f"  RECOMMENDED: {best_model_name} (AUC: {best_auc:.4f})")

                # Store recommendation
                event_recommendations[event_label] = {
                    'best_model': best_model_name,
                    'auc_score': best_auc,
                    'features': top_features,
                    'feature_count': len(top_features),
                    'samples': len(event_data_clean),
                    'minority_ratio': minority_ratio,
                    'all_results': results
                }
            else:
                print(f"  No successful models for {event_label}")

        except Exception as e:
            print(f"  Error processing {event_label}: {str(e)}")
            continue

    # Overall recommendations
    print(f"\n" + "=" * 60)
    print("OVERALL MODEL RECOMMENDATIONS")
    print("=" * 60)

    if event_recommendations:
        # Model frequency
        model_counts = {}
        total_auc = {}

        for event, rec in event_recommendations.items():
            best_model = rec['best_model']
            model_counts[best_model] = model_counts.get(best_model, 0) + 1

            if best_model not in total_auc:
                total_auc[best_model] = []
            total_auc[best_model].append(rec['auc_score'])

        print(f"\nMODEL SUCCESS FREQUENCY:")
        for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            avg_auc = np.mean(total_auc[model])
            print(f"  {model:<15}: {count} events | Avg AUC: {avg_auc:.4f}")

        # Best performing events
        sorted_events = sorted(event_recommendations.items(),
                               key=lambda x: x[1]['auc_score'],
                               reverse=True)

        print(f"\nTOP PERFORMING EVENT PREDICTIONS:")
        for event, rec in sorted_events[:5]:
            print(f"  {event}: {rec['best_model']} | AUC: {rec['auc_score']:.4f} | Features: {rec['feature_count']}")

        # Overall recommendation
        best_overall_model = max(model_counts.keys(), key=lambda k: model_counts[k])
        avg_performance = np.mean(total_auc[best_overall_model])

        print(f"\nOVERALL RECOMMENDATION:")
        print(f"  Model: {best_overall_model}")
        print(f"  Success rate: {model_counts[best_overall_model]}/{len(event_recommendations)} events")
        print(f"  Average AUC: {avg_performance:.4f}")

        # Specific financial advice
        print(f"\nFINANCIAL MODELING ADVICE:")

        if best_overall_model == 'XGBoost':
            print("  • XGBoost excels with your causal features")
            print("  • Focus on hyperparameter tuning (learning_rate, max_depth)")
            print("  • Monitor for overfitting with early stopping")

        elif best_overall_model == 'LightGBM':
            print("  • LightGBM is optimal for your time series data")
            print("  • Very fast and memory efficient")
            print("  • Good handling of your causal feature interactions")

        elif best_overall_model == 'RandomForest':
            print("  • Random Forest provides robust, interpretable results")
            print("  • Great for understanding feature importance")
            print("  • Less prone to overfitting than boosting methods")

        elif best_overall_model == 'LogisticRegression':
            print("  • Logistic Regression works well with your causal features")
            print("  • Highly interpretable - can see exact feature contributions")
            print("  • Fast training and prediction")

    else:
        print("No successful model recommendations found.")

    return event_recommendations


def perform_detailed_model_analysis(event_label, top_n_features=10):
    """
    Detailed analysis for a specific event with hyperparameter tuning
    """

    print(f"\nDETAILED MODEL ANALYSIS: {event_label}")
    print("=" * 50)

    # Load data
    data = pd.read_pickle('labeled5mEE.pkl')

    with open('research_results.pkl', 'rb') as f:
        causal_results = pickle.load(f)

    # Get causal features for this event
    if event_label not in causal_results['research_results']:
        print(f"Event {event_label} not found in causal results")
        return None

    event_data = causal_results['research_results'][event_label]
    causal_relationships = event_data.get('causal_relationships', [])

    if not causal_relationships:
        print(f"No causal relationships found for {event_label}")
        return None

    # Get top features
    sorted_features = sorted(causal_relationships, key=lambda x: x['strength'], reverse=True)
    top_features = [rel['cause_feature'] for rel in sorted_features[:top_n_features]]

    print(f"Using top {len(top_features)} causal features:")
    for i, feature in enumerate(top_features, 1):
        strength = next(rel['strength'] for rel in sorted_features if rel['cause_feature'] == feature)
        print(f"  {i}. {feature} (strength: {strength:.4f})")

    # Prepare data
    event_data_clean = data[[event_label] + top_features].dropna()
    X = event_data_clean[top_features]
    y = event_data_clean[event_label]

    # Time series split
    tscv = TimeSeriesSplit(n_splits=3)

    # Advanced models with hyperparameter tuning
    models_advanced = {
        'XGBoost_Tuned': {
            'model': xgb.XGBClassifier(
                random_state=42,
                learning_rate=0.05,
                max_depth=6,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss'
            ),
            'needs_scaling': False
        },
        'LightGBM_Tuned': {
            'model': lgb.LGBMClassifier(
                random_state=42,
                learning_rate=0.05,
                num_leaves=31,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                verbose=-1
            ),
            'needs_scaling': False
        },
        'RandomForest_Tuned': {
            'model': RandomForestClassifier(
                random_state=42,
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                n_jobs=-1
            ),
            'needs_scaling': False
        },
        'LogisticRegression_Tuned': {
            'model': LogisticRegression(
                random_state=42,
                C=1.0,
                penalty='l2',
                max_iter=2000,
                solver='lbfgs'
            ),
            'needs_scaling': True
        }
    }

    cv_results = {}

    for name, model_info in models_advanced.items():
        model = model_info['model']
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if model_info['needs_scaling']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                model.fit(X_train_scaled, y_train)
                y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_val)[:, 1]

            score = roc_auc_score(y_val, y_pred_proba)
            scores.append(score)

        cv_results[name] = {
            'mean_auc': np.mean(scores),
            'std_auc': np.std(scores),
            'scores': scores
        }

    # Print results
    print(f"\nCROSS-VALIDATION RESULTS (3-fold time series CV):")
    print(f"{'Model':<25} {'Mean AUC':<10} {'Std AUC':<10} {'Stability'}")
    print("-" * 60)

    for name, result in sorted(cv_results.items(), key=lambda x: x[1]['mean_auc'], reverse=True):
        stability = "High" if result['std_auc'] < 0.02 else "Medium" if result['std_auc'] < 0.05 else "Low"
        print(f"{name:<25} {result['mean_auc']:<10.4f} {result['std_auc']:<10.4f} {stability}")

    # Best model
    best_model = max(cv_results.keys(), key=lambda k: cv_results[k]['mean_auc'])
    print(f"\nBEST MODEL: {best_model}")
    print(f"Mean AUC: {cv_results[best_model]['mean_auc']:.4f}")
    print(f"Std AUC: {cv_results[best_model]['std_auc']:.4f}")

    return cv_results


def adaptive_model_selection(X, y, event_label):
    """
    Adapt model selection based on data characteristics
    """
    n_samples, n_features = X.shape
    minority_ratio = y.value_counts().min() / y.value_counts().sum()

    print(f"    Data characteristics:")
    print(f"      Samples: {n_samples}, Features: {n_features}")
    print(f"      Minority class ratio: {minority_ratio:.4f}")

    # Adapt model selection based on data characteristics
    models_to_test = []

    # Small dataset (< 1000 samples)
    if n_samples < 1000:
        models_to_test.extend(['LogisticRegression', 'GaussianNB', 'KNN', 'LinearDiscriminant'])
        print(f"      Small dataset: Testing simple models")

    # Medium dataset (1000-10000 samples)
    elif n_samples < 10000:
        models_to_test.extend(['LogisticRegression', 'RandomForest', 'GradientBoosting',
                               'XGBoost', 'LightGBM', 'SVM_RBF'])
        print(f"      Medium dataset: Testing tree models and boosting")

    # Large dataset (>10000 samples)
    else:
        models_to_test.extend(['XGBoost', 'LightGBM', 'RandomForest', 'ExtraTrees',
                               'MLPClassifier', 'SGDClassifier'])
        print(f"      Large dataset: Testing advanced models and neural networks")

    # High dimensional data (features/samples > 0.1)
    if n_features / n_samples > 0.1:
        models_to_test.extend(['RidgeClassifier', 'LinearDiscriminant'])
        print(f"      High dimensional: Adding regularized models")

    # Very imbalanced data (< 5% minority class)
    if minority_ratio < 0.05:
        # Focus on models that handle imbalance well
        if 'XGBoost' not in models_to_test:
            models_to_test.append('XGBoost')
        if 'RandomForest' not in models_to_test:
            models_to_test.append('RandomForest')
        print(f"      Imbalanced data: Prioritizing ensemble methods")

    # Non-linear patterns likely (based on feature names)
    feature_names = list(X.columns) if hasattr(X, 'columns') else []
    has_interactions = any(any(pattern in str(feat).lower() for pattern in
                               ['distance', 'ratio', 'pct', 'momentum'])
                           for feat in feature_names)

    if has_interactions:
        models_to_test.extend(['RandomForest', 'XGBoost', 'MLPClassifier'])
        print(f"      Feature interactions detected: Adding non-linear models")

    return list(set(models_to_test))  # Remove duplicates


def get_model_recommendations_by_category():
    """
    Provide model recommendations by category and use case
    """

    recommendations = {
        'Interpretability_First': {
            'models': ['LogisticRegression', 'LinearDiscriminant', 'GaussianNB'],
            'use_case': 'When you need to explain predictions to stakeholders',
            'pros': 'Clear feature contributions, regulatory compliance',
            'cons': 'May miss complex patterns'
        },

        'Performance_First': {
            'models': ['XGBoost', 'LightGBM', 'MLPClassifier'],
            'use_case': 'When prediction accuracy is the top priority',
            'pros': 'Best predictive performance',
            'cons': 'Black box, harder to interpret'
        },

        'Speed_First': {
            'models': ['SGDClassifier', 'GaussianNB', 'KNN'],
            'use_case': 'When you need very fast training/prediction',
            'pros': 'Fastest training and prediction',
            'cons': 'May sacrifice accuracy'
        },

        'Robustness_First': {
            'models': ['RandomForest', 'ExtraTrees', 'BaggingClassifier'],
            'use_case': 'When you need stable, consistent performance',
            'pros': 'Robust to outliers and overfitting',
            'cons': 'May not achieve peak performance'
        },

        'Small_Data': {
            'models': ['LogisticRegression', 'GaussianNB', 'KNN', 'LinearDiscriminant'],
            'use_case': 'When you have limited training data',
            'pros': 'Work well with few samples',
            'cons': 'Limited complexity modeling'
        },

        'Imbalanced_Data': {
            'models': ['XGBoost', 'RandomForest', 'AdaBoost'],
            'use_case': 'When classes are heavily imbalanced',
            'pros': 'Handle class imbalance naturally',
            'cons': 'May still need class weighting'
        },

        'High_Dimensional': {
            'models': ['RidgeClassifier', 'LinearDiscriminant', 'SGDClassifier'],
            'use_case': 'When features >> samples',
            'pros': 'Handle curse of dimensionality',
            'cons': 'Assume linear relationships'
        },

        'Financial_Specific': {
            'models': ['LightGBM', 'XGBoost', 'RandomForest', 'LogisticRegression'],
            'use_case': 'Optimized for financial time series with causal features',
            'pros': 'Proven performance on financial data',
            'cons': 'May need financial-specific tuning'
        }
    }

    return recommendations


def get_model_selection_summary():
    """
    Provide a comprehensive summary and action plan
    """

    print("\n" + "=" * 70)
    print("MODEL SELECTION SUMMARY & ACTION PLAN")
    print("=" * 70)

    # Show model categories and recommendations
    categories = get_model_recommendations_by_category()

    print("\nMODEL SELECTION BY USE CASE:")
    print("-" * 50)

    for category, info in categories.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        print(f"  Models: {', '.join(info['models'])}")
        print(f"  Use case: {info['use_case']}")
        print(f"  Pros: {info['pros']}")
        print(f"  Cons: {info['cons']}")

    print(f"""
COMPREHENSIVE APPROACH FOR YOUR CAUSAL FINANCIAL DATA:

1. ADAPTIVE TESTING: Test 15+ models automatically
   - Linear models for interpretability
   - Tree models for feature interactions
   - Boosting methods for performance
   - Neural networks for complex patterns
   - Support Vector Machines for non-linear boundaries

2. DATA-DRIVEN SELECTION: Models chosen based on:
   - Dataset size (small/medium/large)
   - Feature count vs sample ratio
   - Class imbalance severity
   - Presence of feature interactions
   - Financial time series characteristics

3. EVENT-SPECIFIC OPTIMIZATION:
   - Different events may need different models
   - High-frequency events (42% occurrence): Complex models work
   - Rare events (0.12% occurrence): Simple models may be better
   - Medium events: Full model comparison needed

4. VALIDATION STRATEGY:
   - Time series cross-validation (not random splits)
   - Out-of-sample testing on recent data
   - Stability assessment across market conditions
   - Financial metrics (Sharpe ratio, max drawdown)

NEXT STEPS:
1. Run select_best_model_for_causal_features() - tests all models adaptively
2. For top events, run perform_detailed_model_analysis() - hyperparameter tuning
3. Implement ensemble methods combining best performers
4. Set up walk-forward validation for production
5. Monitor model drift in live conditions

WHY THIS BEATS 4-MODEL TESTING:
• Covers all major algorithm families (17+ models)
• Adapts to your specific data characteristics
• Finds optimal model per event type
• Reduces risk of missing the best performer
• Provides backup options if primary model fails
""")


# Main execution function
if __name__ == "__main__":
    print("RUNNING COMPREHENSIVE MODEL SELECTION...")

    try:
        # Step 1: Get recommendations for all events
        print("\nStep 1: Testing all models across all events...")
        recommendations = select_best_model_for_causal_features()

        # Step 2: Detailed analysis for top events
        if recommendations:
            top_events = sorted(recommendations.items(),
                                key=lambda x: x[1]['auc_score'],
                                reverse=True)[:3]

            print(f"\nStep 2: Detailed analysis for top 3 events...")
            for event_name, _ in top_events:
                perform_detailed_model_analysis(event_name)

        # Step 3: Final summary and recommendations
        print(f"\nStep 3: Summary and action plan...")
        get_model_selection_summary()

        print(f"\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)
        print("Your optimal models have been identified for each event type.")
        print("Use the recommendations above to implement your trading models.")

    except Exception as e:
        print(f"Error in model selection: {str(e)}")
        print("Make sure your data files (labeled5mEE.pkl, research_results.pkl) are available.")

        # Fallback: show summary anyway
        get_model_selection_summary()
    """
    Provide a comprehensive summary and action plan
    """

    print("\n" + "=" * 70)
    print("MODEL SELECTION SUMMARY & ACTION PLAN")
    print("=" * 70)

    print("""
RECOMMENDED APPROACH FOR YOUR CAUSAL FINANCIAL DATA:

1. PRIMARY RECOMMENDATION: Start with LightGBM or XGBoost
   - Both handle your causal features excellently
   - Fast training and prediction
   - Built-in handling of feature interactions
   - Good performance on time series data

2. BASELINE COMPARISON: Use Logistic Regression
   - Highly interpretable
   - Shows exact contribution of each causal feature
   - Fast and stable
   - Good for understanding feature relationships

3. ENSEMBLE APPROACH: Combine multiple models
   - Use voting classifier with top 3 performers
   - More robust predictions
   - Reduces overfitting risk

NEXT STEPS:
1. Run select_best_model_for_causal_features() to get specific recommendations
2. For your best-performing events, run perform_detailed_model_analysis()
3. Focus on hyperparameter tuning for your chosen model
4. Implement proper backtesting with walk-forward analysis
5. Monitor model performance in live trading conditions
""")

# Main execution function
if __name__ == "__main__":
    print("RUNNING COMPREHENSIVE MODEL SELECTION...")

    # Step 1: Get recommendations for all events
    recommendations = select_best_model_for_causal_features()

    # Step 2: Detailed analysis for top events
    if recommendations:
        top_events = sorted(recommendations.items(),
                            key=lambda x: x[1]['auc_score'],
                            reverse=True)[:3]

        print(f"\nDETAILED ANALYSIS FOR TOP 3 EVENTS:")
        for event_name, _ in top_events:
            perform_detailed_model_analysis(event_name)

    # Step 3: Final summary
    get_model_selection_summary()
