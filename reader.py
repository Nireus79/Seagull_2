import pickle
import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_pickle('D:/Seagull_data/labeled5mEE2cov.pkl')
for i in df.columns:
    print(i)
#
#
# def analyze_causal_research_results(results_file='research_results.pkl'):
#     """Analyze and display detailed causal research results"""
#
#     try:
#         # Load the results
#         with open(results_file, 'rb') as f:
#             results = pickle.load(f)
#
#         print("=" * 80)
#         print("DETAILED CAUSAL RESEARCH ANALYSIS")
#         print("=" * 80)
#
#         # Overall statistics
#         research_results = results.get('research_results', {})
#         print(f"\nOVERALL STATISTICS:")
#         print(f"Total events researched: {len(research_results)}")
#
#         causal_analysis = results.get('causal_analysis', {})
#         if isinstance(causal_analysis, dict) and 'summary' in causal_analysis:
#             summary = causal_analysis['summary']
#             print(f"Total relationships found: {summary.get('total_relationships', 'N/A')}")
#             print(f"Average causal strength: {summary.get('average_causal_strength', 'N/A')}")
#
#         print("\n" + "=" * 60)
#         print("DETAILED CAUSAL RELATIONSHIPS BY EVENT")
#         print("=" * 60)
#
#         # Analyze each event
#         for event_label, event_data in research_results.items():
#             print(f"\n🎯 EVENT: {event_label}")
#             print("-" * 50)
#
#             # Basic event info
#             occurrences = event_data.get('event_occurrences', 'N/A')
#             frequency = event_data.get('event_frequency', 'N/A')
#             print(f"Occurrences: {occurrences}")
#             print(f"Frequency: {frequency:.4f}" if isinstance(frequency, (int, float)) else f"Frequency: {frequency}")
#
#             # Individual causal relationships
#             relationships = event_data.get('causal_relationships', [])
#             print(f"\nCAUSAL RELATIONSHIPS FOUND: {len(relationships)}")
#
#             if relationships:
#                 print("\nTop Causal Features:")
#                 # Sort by strength
#                 sorted_relationships = sorted(relationships,
#                                               key=lambda x: x.get('strength', 0),
#                                               reverse=True)
#
#                 for i, rel in enumerate(sorted_relationships[:10], 1):  # Top 10
#                     cause = rel.get('cause_feature', 'Unknown')
#                     strength = rel.get('strength', 0)
#                     p_value = rel.get('p_value', 1.0)
#                     mechanism = rel.get('economic_mechanism', 'Unknown')
#                     justification = rel.get('economic_justification', 'No justification')
#
#                     print(f"  {i:2d}. {cause}")
#                     print(f"      Strength: {strength:.4f} | P-value: {p_value:.6f}")
#                     print(f"      Mechanism: {mechanism}")
#                     print(f"      Justification: {justification}")
#                     print()
#
#             # Validation results if available
#             validation_results = event_data.get('validation_results', {})
#             if validation_results:
#                 print(f"VALIDATION RESULTS:")
#                 for relationship_key, validation in validation_results.items():
#                     auc_roc = validation.get('auc_roc', 'N/A')
#                     sharpe_ratio = validation.get('sharpe_ratio', 'N/A')
#                     total_return = validation.get('total_return', 'N/A')
#
#                     print(f"  {relationship_key}:")
#                     print(f"    AUC-ROC: {auc_roc:.4f}" if isinstance(auc_roc,
#                                                                       (int, float)) else f"    AUC-ROC: {auc_roc}")
#                     print(f"    Sharpe Ratio: {sharpe_ratio:.4f}" if isinstance(sharpe_ratio, (
#                         int, float)) else f"    Sharpe Ratio: {sharpe_ratio}")
#                     print(f"    Total Return: {total_return:.4f}" if isinstance(total_return, (
#                         int, float)) else f"    Total Return: {total_return}")
#
#             print("\n" + "-" * 50)
#
#         # Feature importance analysis
#         print("\n" + "=" * 60)
#         print("FEATURE IMPORTANCE ANALYSIS")
#         print("=" * 60)
#
#         # Collect all causal relationships
#         all_features = {}
#         for event_label, event_data in research_results.items():
#             relationships = event_data.get('causal_relationships', [])
#             for rel in relationships:
#                 feature = rel.get('cause_feature', 'Unknown')
#                 strength = rel.get('strength', 0)
#
#                 if feature not in all_features:
#                     all_features[feature] = []
#                 all_features[feature].append({
#                     'event': event_label,
#                     'strength': strength,
#                     'p_value': rel.get('p_value', 1.0)
#                 })
#
#         # Sort features by average strength
#         feature_importance = []
#         for feature, occurrences in all_features.items():
#             avg_strength = np.mean([occ['strength'] for occ in occurrences])
#             count = len(occurrences)
#             feature_importance.append({
#                 'feature': feature,
#                 'avg_strength': avg_strength,
#                 'count': count,
#                 'occurrences': occurrences
#             })
#
#         feature_importance.sort(key=lambda x: x['avg_strength'], reverse=True)
#
#         print(f"\nTOP 15 MOST IMPORTANT CAUSAL FEATURES:")
#         print(f"{'Rank':>4} {'Feature':>25} {'Avg Strength':>12} {'Count':>6} {'Events'}")
#         print("-" * 70)
#
#         for i, feat in enumerate(feature_importance[:15], 1):
#             events_str = ", ".join([occ['event'].split('_')[0] for occ in feat['occurrences'][:3]])
#             if len(feat['occurrences']) > 3:
#                 events_str += f" (+{len(feat['occurrences']) - 3} more)"
#
#             print(f"{i:4d} {feat['feature']:>25} {feat['avg_strength']:>12.4f} {feat['count']:>6d} {events_str}")
#
#         # Economic mechanism analysis
#         print(f"\n" + "=" * 60)
#         print("ECONOMIC MECHANISM ANALYSIS")
#         print("=" * 60)
#
#         mechanism_counts = {}
#         for event_label, event_data in research_results.items():
#             relationships = event_data.get('causal_relationships', [])
#             for rel in relationships:
#                 mechanism = rel.get('economic_mechanism', 'Unknown')
#                 if mechanism not in mechanism_counts:
#                     mechanism_counts[mechanism] = 0
#                 mechanism_counts[mechanism] += 1
#
#         sorted_mechanisms = sorted(mechanism_counts.items(), key=lambda x: x[1], reverse=True)
#
#         print(f"\nMechanism distribution:")
#         for mechanism, count in sorted_mechanisms:
#             percentage = (count / sum(mechanism_counts.values())) * 100
#             print(f"  {mechanism}: {count} relationships ({percentage:.1f}%)")
#
#         return results
#
#     except FileNotFoundError:
#         print(f"Results file '{results_file}' not found.")
#         print("Please make sure the research results have been saved to this file.")
#         return None
#     except Exception as e:
#         print(f"Error analyzing results: {str(e)}")
#         return None
#
#
# def extract_best_features_by_event(results_file='research_results.pkl', top_n=5):
#     """Extract the best causal features for each event"""
#
#     try:
#         with open(results_file, 'rb') as f:
#             results = pickle.load(f)
#
#         research_results = results.get('research_results', {})
#         best_features = {}
#
#         for event_label, event_data in research_results.items():
#             relationships = event_data.get('causal_relationships', [])
#
#             if relationships:
#                 # Sort by strength
#                 sorted_rels = sorted(relationships,
#                                      key=lambda x: x.get('strength', 0),
#                                      reverse=True)
#
#                 best_features[event_label] = []
#                 for rel in sorted_rels[:top_n]:
#                     best_features[event_label].append({
#                         'feature': rel.get('cause_feature', 'Unknown'),
#                         'strength': rel.get('strength', 0),
#                         'p_value': rel.get('p_value', 1.0),
#                         'mechanism': rel.get('economic_mechanism', 'Unknown')
#                     })
#
#         return best_features
#
#     except Exception as e:
#         print(f"Error extracting features: {str(e)}")
#         return None
#
#
# def identify_column_types(data):
#     """Identify different types of columns in the dataset"""
#
#     # Event columns (contain 'event' or 'covent' in name, boolean values)
#     event_columns = []
#     for col in data.columns:
#         if ('event' in col.lower() or 'covent' in col.lower()) and not col.endswith('_label'):
#             # Check if column contains boolean-like values
#             unique_vals = data[col].dropna().unique()
#             if len(unique_vals) > 0 and set(unique_vals).issubset({True, False, 0, 1, 0.0, 1.0}):
#                 event_columns.append(col)
#
#     # Label columns (contain 'label' in name)
#     label_columns = [col for col in data.columns if col.endswith('_label')]
#
#     # Technical indicator columns (numeric, not event/label/ohlcv)
#     exclude_patterns = {'_label', '_barrier_touched', '_touch_time', '_return', '_holding_hours',
#                         '_event', 'event_type', 'any_event'}
#     ohlcv_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
#     numeric_dtypes = {'float64', 'int64', 'float32', 'int32', 'float16', 'int16'}
#
#     technical_indicator_columns = []
#     for col in data.columns:
#         # Skip if it's an event or label column
#         if col in event_columns or col in label_columns:
#             continue
#         # Skip if it matches exclude patterns
#         if any(pattern in col.lower() for pattern in exclude_patterns):
#             continue
#         # Include if numeric and not completely empty
#         if data[col].dtype.name in numeric_dtypes or col in ohlcv_cols:
#             if not data[col].isna().all():
#                 technical_indicator_columns.append(col)
#
#     return {
#         'event_columns': event_columns,
#         'label_columns': label_columns,
#         'technical_indicator_columns': technical_indicator_columns
#     }
#
#
# def data_stats(data):
#     event_columns, label_columns, feature_columns = identify_column_types(data)
#     """Print data statistics"""
#     print("Event column analysis:")
#     for col in event_columns[:5]:
#         if col in data.columns:
#             true_events = ((data[col] is True) | (data[col] == 1)).sum()
#             total_rows = len(data)
#             print(f"{col}: {true_events} events ({true_events / total_rows:.3%} of data)")
#
#     print("\nLabel distributions:")
#     for col in label_columns[:5]:
#         if col in data.columns:
#             print(f"{col}: {data[col].value_counts().to_dict()}")
#
#     print(f"\nFeature statistics:")
#     print(f"Technical indicators: {len(feature_columns)}")
#     # if feature_columns:
#     #     feature_completeness = (1 - data[feature_columns].isna().mean()).mean()
#     #     print(f"Average feature completeness: {feature_completeness:.1%}")
#
#
# # Usage example
# if __name__ == "__main__":
#     # Analyze the research results
#     d = pd.read_pickle('D:/Seagull_data/labeled5mEE2cov.pkl')
#     results = analyze_causal_research_results('causal_results/research_results.pkl')
#
#     data_stats(d)
#     # Extract best features for each event
#     print(f"\n" + "=" * 60)
#     print("QUICK REFERENCE: BEST FEATURES BY EVENT")
#     print("=" * 60)
#
#     best_features = extract_best_features_by_event('causal_results/research_results.pkl', top_n=5)
#     if best_features:
#         for event, features in best_features.items():
#             print(f"\n{event}:")
#             for i, feat in enumerate(features, 1):
#                 print(f"  {i}. {feat['feature']} (strength: {feat['strength']:.4f})")
