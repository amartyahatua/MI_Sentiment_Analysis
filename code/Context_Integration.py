import torch
import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import os

my_seed = random.randint(1, 100)
# my_seed = 6
print('Seed = ', my_seed)

random.seed(my_seed)
# Set seed for NumPy's random number generator
np.random.seed(my_seed)
# Set seed for PyTorch's CPU random number generator
torch.manual_seed(my_seed)
# Set seed for PyTorch's CUDA (GPU) random number generator
torch.cuda.manual_seed(my_seed)
# Set seed for all PyTorch's CUDA (GPU) random number generators (for multi-GPU setups)
torch.cuda.manual_seed_all(my_seed)
# Set a fixed value for the hash seed (affects Python's hash functions, important for some data structures)
os.environ['PYTHONHASHSEED'] = str(my_seed)

class AllLayerContextualAnalyzer:
    """
    Comprehensive analysis of contextual processing across ALL layers without bias
    """

    def __init__(self, model_name: str = "gpt2"):
        self.model = HookedTransformer.from_pretrained(model_name)
        self.model.eval()
        self.n_layers = self.model.cfg.n_layers
        self.d_model = self.model.cfg.d_model
        print(f"Initialized model with {self.n_layers} layers")

    def generate_test_cases(self) -> Dict[str, List[Dict]]:
        """Generate diverse contextual test cases"""
        context_list = ['Intensity',  'Scale variation', 'Sarcasm', 'Domain context', 'Multiple intensifier', 'Complex double negation']

        df_cases = pd.read_csv('../data/sentiment_2000_pairs.csv')
        df_cases = df_cases[['clean_text', 'corrupt_text', 'Context']]
        cases = {}

        for context in context_list:
            df_cases_context = df_cases[df_cases['Context'] == context]
            context_data = []
            for i in range(df_cases_context.shape[0]):
                data = df_cases_context.iloc[i]
                context_data.append(
                    {
                        'baseline':data['clean_text'],
                        'contextual': data['corrupt_text']
                    }
                )

            cases[context] = context_data


        return cases

    def get_all_layer_activations(self, text: str) -> Dict[str, np.ndarray]:
        """Extract comprehensive activations from all layers"""

        with torch.no_grad():
            tokens = self.model.to_tokens(text)
            logits, cache = self.model.run_with_cache(tokens)

            # Initialize storage for different activation types
            activation_types = {
                'residual_post': np.zeros(self.n_layers),
                'residual_mid': np.zeros(self.n_layers),
                'attention_out': np.zeros(self.n_layers),
                'mlp_out': np.zeros(self.n_layers),
                'attention_scores': np.zeros(self.n_layers),
                'mlp_pre': np.zeros(self.n_layers)
            }

            for layer in range(self.n_layers):
                # 1. Residual stream after each block
                resid_post_key = f"blocks.{layer}.hook_resid_post"
                if resid_post_key in cache:
                    resid_post = cache[resid_post_key][0]  # [seq_len, d_model]
                    activation_types['residual_post'][layer] = resid_post.mean(dim=0).norm().item()

                # 2. Residual stream in middle of block
                resid_mid_key = f"blocks.{layer}.hook_resid_mid"
                if resid_mid_key in cache:
                    resid_mid = cache[resid_mid_key][0]
                    activation_types['residual_mid'][layer] = resid_mid.mean(dim=0).norm().item()

                # 3. Attention output
                attn_out_key = f"blocks.{layer}.hook_attn_out"
                if attn_out_key in cache:
                    attn_out = cache[attn_out_key][0]
                    activation_types['attention_out'][layer] = attn_out.mean(dim=0).norm().item()

                # 4. MLP output
                mlp_out_key = f"blocks.{layer}.hook_mlp_out"
                if mlp_out_key in cache:
                    mlp_out = cache[mlp_out_key][0]
                    activation_types['mlp_out'][layer] = mlp_out.mean(dim=0).norm().item()

                # 5. Attention pattern entropy
                attn_pattern_key = f"blocks.{layer}.attn.hook_attn_scores"
                if attn_pattern_key in cache:
                    attn_patterns = cache[attn_pattern_key][0]  # [n_heads, seq_len, seq_len]
                    # Calculate average entropy across heads
                    entropies = []
                    for head in range(attn_patterns.shape[0]):
                        pattern = attn_patterns[head].flatten()
                        pattern = torch.softmax(pattern, dim=0)
                        entropy = -(pattern * torch.log(pattern + 1e-10)).sum().item()
                        entropies.append(entropy)
                    activation_types['attention_scores'][layer] = np.mean(entropies)

                # 6. MLP pre-activation
                mlp_pre_key = f"blocks.{layer}.mlp.hook_pre"
                if mlp_pre_key in cache:
                    mlp_pre = cache[mlp_pre_key][0]
                    activation_types['mlp_pre'][layer] = mlp_pre.mean(dim=0).norm().item()

        return activation_types

    def calculate_layer_differences(self, baseline_activations: Dict[str, np.ndarray],
                                  contextual_activations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate differences between baseline and contextual activations"""

        differences = {}

        for activation_type in baseline_activations.keys():
            baseline = baseline_activations[activation_type]
            contextual = contextual_activations[activation_type]

            # Multiple difference metrics
            abs_diff = np.abs(contextual - baseline)
            rel_diff = np.abs(contextual - baseline) / (np.abs(baseline) + 1e-8)
            squared_diff = (contextual - baseline) ** 2

            # Combine metrics (you can experiment with different combinations)
            combined_diff = abs_diff + 0.5 * rel_diff + 0.3 * squared_diff

            differences[activation_type] = combined_diff

        return differences

    def analyze_all_layers(self, cases: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Comprehensive analysis across all layers and activation types"""

        print("Starting comprehensive all-layer analysis...")
        print("=" * 70)

        results = {}

        for context_type, case_list in cases.items():
            print(f"\n🔍 ANALYZING {context_type.upper()}")
            print("-" * 50)

            # Store results for each activation type
            activation_results = {
                'residual_post': [],
                'residual_mid': [],
                'attention_out': [],
                'mlp_out': [],
                'attention_scores': [],
                'mlp_pre': []
            }

            case_summaries = []

            # Process each case
            for i, case in enumerate(tqdm(case_list, desc=f"Processing {context_type}")):
                # Get activations for both versions
                baseline_acts = self.get_all_layer_activations(case['baseline'])
                contextual_acts = self.get_all_layer_activations(case['contextual'])

                # Calculate differences
                differences = self.calculate_layer_differences(baseline_acts, contextual_acts)

                # Store for aggregation
                for act_type, diff in differences.items():
                    activation_results[act_type].append(diff)

                # Find peak layer for this case
                combined_effect = np.mean(list(differences.values()), axis=0)
                peak_layer = np.argmax(combined_effect)
                peak_strength = combined_effect[peak_layer]

                case_summaries.append({
                    'case_num': i + 1,
                    'peak_layer': peak_layer,
                    'peak_strength': peak_strength,
                    'baseline': case['baseline'] + "...",
                    'contextual': case['contextual'] + "..."
                })

                print(f"  Case {i+1}: Peak at Layer {peak_layer} (strength: {peak_strength:.3f})")

            # Aggregate results across all cases
            aggregated_results = {}
            for act_type, all_diffs in activation_results.items():
                avg_diff = np.mean(all_diffs, axis=0)
                std_diff = np.std(all_diffs, axis=0)

                peak_layer = np.argmax(avg_diff)
                peak_strength = avg_diff[peak_layer]

                aggregated_results[act_type] = {
                    'avg_effects': avg_diff,
                    'std_effects': std_diff,
                    'peak_layer': peak_layer,
                    'peak_strength': peak_strength
                }

            # Find consensus across activation types
            consensus_analysis = self.find_consensus_layers(aggregated_results)

            # Store comprehensive results
            results[context_type] = {
                'activation_results': aggregated_results,
                'consensus': consensus_analysis,
                'case_summaries': case_summaries,
                'strongest_layers': self.find_strongest_layers(aggregated_results),
                'layer_rankings': self.rank_all_layers(aggregated_results)
            }

            # Print summary
            print(f"\n📊 {context_type.upper()} SUMMARY:")
            print(f"  Consensus peak layer: {consensus_analysis['consensus_layer']}")
            print(f"  Agreement strength: {consensus_analysis['agreement_score']:.3f}")
            print(f"  Top 3 layers: {consensus_analysis['top_3_layers']}")

        return results

    def find_consensus_layers(self, activation_results: Dict[str, Dict]) -> Dict:
        """Find consensus across different activation types"""

        # Get peak layer for each activation type
        peak_layers = []
        for act_type, result in activation_results.items():
            peak_layers.append(result['peak_layer'])

        # Find most common peak
        from collections import Counter
        layer_counts = Counter(peak_layers)
        consensus_layer = layer_counts.most_common(1)[0][0]
        agreement_score = layer_counts[consensus_layer] / len(peak_layers)

        # Get top 3 layers across all activation types
        all_effects = np.zeros(self.n_layers)
        for result in activation_results.values():
            all_effects += result['avg_effects']

        top_3_layers = np.argsort(all_effects)[-3:][::-1].tolist()

        return {
            'consensus_layer': consensus_layer,
            'agreement_score': agreement_score,
            'top_3_layers': top_3_layers,
            'layer_vote_counts': dict(layer_counts),
            'combined_effects': all_effects
        }

    def find_strongest_layers(self, activation_results: Dict[str, Dict]) -> List[Tuple[int, float]]:
        """Find the strongest layers overall"""

        # Combine effects across all activation types
        combined_effects = np.zeros(self.n_layers)
        for result in activation_results.values():
            combined_effects += result['avg_effects']

        # Get top 5 strongest layers
        top_indices = np.argsort(combined_effects)[-5:][::-1]
        return [(int(layer), float(combined_effects[layer])) for layer in top_indices]

    def rank_all_layers(self, activation_results: Dict[str, Dict]) -> np.ndarray:
        """Rank all layers by their average contextual sensitivity"""

        combined_effects = np.zeros(self.n_layers)
        for result in activation_results.values():
            # Normalize each activation type to [0,1] before combining
            normalized_effects = result['avg_effects'] / (np.max(result['avg_effects']) + 1e-8)
            combined_effects += normalized_effects

        # Convert to rankings (0 = strongest, n_layers-1 = weakest)
        rankings = np.argsort(np.argsort(combined_effects)[::-1])
        return rankings

    def create_comprehensive_visualizations(self, results: Dict[str, Any]) -> None:
        """Create comprehensive visualizations for all-layer analysis"""

        # Create a large figure with multiple subplots
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('Comprehensive All-Layer Contextual Analysis', fontsize=16, fontweight='bold')

        context_types = list(results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(context_types)))

        # 1. Consensus peak layers
        consensus_layers = [results[ctx]['consensus']['consensus_layer'] for ctx in context_types]
        agreement_scores = [results[ctx]['consensus']['agreement_score'] for ctx in context_types]

        bars = axes[0,0].bar(context_types, consensus_layers, color=colors, alpha=0.7)
        axes[0,0].set_title('Consensus Peak Layers')
        axes[0,0].set_ylabel('Layer Number')
        axes[0,0].tick_params(axis='x', rotation=45)

        # Add agreement scores as text on bars
        for bar, score in zip(bars, agreement_scores):
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                          f'{score:.2f}', ha='center', va='bottom', fontsize=8)

        # 2. Layer effect heatmap across all contexts
        heatmap_data = []
        for ctx in context_types:
            combined_effects = results[ctx]['consensus']['combined_effects']
            heatmap_data.append(combined_effects)

        sns.heatmap(heatmap_data, xticklabels=range(self.n_layers),
                   yticklabels=context_types, cmap='viridis', ax=axes[0,1])
        axes[0,1].set_title('Layer Effects Heatmap')
        axes[0,1].set_xlabel('Layer')

        # 3. Top 3 layers distribution
        all_top3 = []
        for ctx in context_types:
            all_top3.extend(results[ctx]['consensus']['top_3_layers'])

        layer_importance = np.bincount(all_top3, minlength=self.n_layers)
        axes[0,2].bar(range(self.n_layers), layer_importance, color='skyblue', alpha=0.7)
        axes[0,2].set_title('Layer Importance (Top-3 Frequency)')
        axes[0,2].set_xlabel('Layer')
        axes[0,2].set_ylabel('Frequency in Top-3')

        # 4. Agreement score distribution
        axes[0,3].hist(agreement_scores, bins=10, color='lightcoral', alpha=0.7, edgecolor='black')
        axes[0,3].set_title('Agreement Score Distribution')
        axes[0,3].set_xlabel('Agreement Score')
        axes[0,3].set_ylabel('Frequency')

        # 5-8. Individual context type detailed plots
        for i, ctx in enumerate(context_types[:4]):
            if i < 4:  # Only plot first 4 context types in detail
                row = 1 + i // 2
                col = i % 2

                # Plot layer effects for all activation types
                act_results = results[ctx]['activation_results']
                for act_type, result in act_results.items():
                    axes[row, col].plot(range(self.n_layers), result['avg_effects'],
                                      label=act_type, marker='o', markersize=3, alpha=0.7)

                axes[row, col].set_title(f'{ctx.capitalize()} - All Activation Types')
                axes[row, col].set_xlabel('Layer')
                axes[row, col].set_ylabel('Effect Magnitude')
                axes[row, col].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                axes[row, col].grid(True, alpha=0.3)

        # 9. Overall layer ranking
        all_rankings = np.zeros(self.n_layers)
        for ctx in context_types:
            rankings = results[ctx]['layer_rankings']
            all_rankings += rankings

        avg_rankings = all_rankings / len(context_types)
        sorted_layers = np.argsort(avg_rankings)

        axes[2,2].bar(range(self.n_layers), avg_rankings[sorted_layers],
                     color='gold', alpha=0.7)
        axes[2,2].set_title('Average Layer Rankings (Lower = More Important)')
        axes[2,2].set_xlabel('Layer (sorted by importance)')
        axes[2,2].set_ylabel('Average Ranking')
        axes[2,2].set_xticks(range(0, self.n_layers, 2))
        axes[2,2].set_xticklabels(sorted_layers[::2])

        # 10. Strongest layers summary
        strongest_summary = {}
        for ctx in context_types:
            for layer, strength in results[ctx]['strongest_layers'][:3]:
                strongest_summary[layer] = strongest_summary.get(layer, 0) + strength

        if strongest_summary:
            layers, strengths = zip(*sorted(strongest_summary.items(), key=lambda x: x[1], reverse=True)[:10])
            axes[2,3].bar(range(len(layers)), strengths, color='purple', alpha=0.7)
            axes[2,3].set_title('Top 10 Strongest Layers Overall')
            axes[2,3].set_xlabel('Rank')
            axes[2,3].set_ylabel('Combined Strength')
            axes[2,3].set_xticks(range(len(layers)))
            axes[2,3].set_xticklabels([f'L{layer}' for layer in layers], rotation=45)

        plt.tight_layout()
        plt.savefig('comprehensive_all_layer_analysis.png', dpi=300, bbox_inches='tight')
        # plt.show()
        # plt.savefig('Second_Phase_New_Code.png')
    def print_final_summary(self, results: Dict[str, Any]) -> None:
        """Print comprehensive final summary"""

        print("\n" + "="*80)
        print("🎯 FINAL ALL-LAYER ANALYSIS SUMMARY")
        print("="*80)

        # Overall patterns
        all_consensus_layers = []
        all_agreement_scores = []

        for ctx, result in results.items():
            all_consensus_layers.append(result['consensus']['consensus_layer'])
            all_agreement_scores.append(result['consensus']['agreement_score'])

        print(f"\n📈 OVERALL PATTERNS:")
        print(f"  Most common peak layer: {max(set(all_consensus_layers), key=all_consensus_layers.count)}")
        print(f"  Average agreement score: {np.mean(all_agreement_scores):.3f}")
        print(f"  Layer range: {min(all_consensus_layers)} - {max(all_consensus_layers)}")

        # Context-specific insights
        print(f"\n🔍 CONTEXT-SPECIFIC INSIGHTS:")
        for ctx, result in results.items():
            consensus = result['consensus']
            print(f"  {ctx.upper()}:")
            print(f"    Peak layer: {consensus['consensus_layer']}")
            print(f"    Top 3 layers: {consensus['top_3_layers']}")
            print(f"    Agreement: {consensus['agreement_score']:.3f}")

        # Most important layers across all contexts
        layer_importance = np.zeros(self.n_layers)
        for result in results.values():
            layer_importance += result['consensus']['combined_effects']

        top_5_layers = np.argsort(layer_importance)[-5:][::-1]
        print(f"\n🏆 TOP 5 MOST IMPORTANT LAYERS OVERALL:")
        for i, layer in enumerate(top_5_layers):
            print(f"  {i+1}. Layer {layer}: {layer_importance[layer]:.3f}")

        print(f"\n💡 KEY FINDINGS:")
        print(f"  • Early layers (0-3): {np.sum(layer_importance[:4]):.1f} total importance")
        print(f"  • Mid layers (4-7): {np.sum(layer_importance[4:8]):.1f} total importance")
        print(f"  • Late layers (8-11): {np.sum(layer_importance[8:]):.1f} total importance")

    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run the complete all-layer analysis"""

        print("🚀 Starting Complete All-Layer Contextual Analysis")
        print("="*70)

        # Generate test cases
        cases = self.generate_test_cases()

        # Run comprehensive analysis
        results = self.analyze_all_layers(cases)

        # Create visualizations
        self.create_comprehensive_visualizations(results)

        # Print final summary
        self.print_final_summary(results)

        return results


# Usage function
def run_all_layer_analysis():
    """Run the complete all-layer analysis"""

    analyzer = AllLayerContextualAnalyzer("gpt2")
    results = analyzer.run_complete_analysis()

    return analyzer, results


if __name__ == "__main__":
    analyzer, results = run_all_layer_analysis()
    print("\n✅ All-layer analysis complete!")
    print("📊 Check 'comprehensive_all_layer_analysis.png' for detailed visualizations")