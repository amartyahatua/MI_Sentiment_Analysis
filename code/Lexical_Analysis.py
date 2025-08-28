# STAGE 1: LEXICAL DETECTION (LAYERS 0-3) - IMPLEMENTATION & TESTING
# Complete implementation guide for testing early layer sentiment processing

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_lens import HookedTransformer
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import os
import pickle
from sklearn.linear_model import LogisticRegression
from transformers import pipeline
import random
from tqdm import  tqdm


# my_seed = random.randint(1, 100)
my_seed = 6
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
# =============================================================================
# 1. DATA STRUCTURES FOR LEXICAL TESTING
# =============================================================================

@dataclass
class LexicalPair:
    """Structure for lexical sentiment pairs"""
    clean_text: str
    corrupt_text: str
    target_word_clean: str  # The sentiment word in clean text
    target_word_corrupt: str  # The sentiment word in corrupt text
    target_position: int  # Token position of target word
    expected_sentiment_flip: bool  # Should sentiment flip when patched?
    clean_sentiment: float  # Expected sentiment score for clean
    corrupt_sentiment: float  # Expected sentiment score for corrupt


class LexicalTestSuite:
    """Generate and manage lexical test cases"""

    def __init__(self):
        self.positive_words = [
            "amazing", "fantastic", "excellent", "wonderful", "brilliant",
            "outstanding", "superb", "magnificent", "terrific", "awesome",
            "delightful", "marvelous", "incredible", "spectacular", "perfect"
        ]

        self.negative_words = [
            "terrible", "awful", "horrible", "disgusting", "pathetic",
            "dreadful", "disappointing", "mediocre", "boring", "annoying",
            "frustrating", "irritating", "unpleasant", "disturbing", "tragic"
        ]

        self.neutral_contexts = [
            "The movie was {}",
            "This book is {}",
            "The restaurant was {}",
            "The service was {}",
            "The experience was {}",
            "I found it {}",
            "It was absolutely {}",
            "Overall, it was {}"
        ]

        self.intensifiers = ["very", "extremely", "absolutely", "quite", "rather"]

    def generate_lexical_pairs(self, num_pairs: int = 200) -> List[LexicalPair]:
        """Generate clean/corrupt pairs for lexical testing"""
        pairs = []
        data = pd.read_csv('./sentiment_2000_pairs.csv')
        # Simple positive/negative swaps
        data = data.iloc[0:200,:]
        for i in range(data.shape[0]):
            temp_data = data.iloc[i]
            pos_word = temp_data['target_tokens'].split(',')[0].strip()
            neg_word = temp_data['target_tokens'].split(',')[1].strip()

            clean_text = temp_data['clean_text']
            corrupt_text = temp_data['corrupt_text']

            target_pos = int(temp_data['target_positions'])

            if temp_data['difficulty'] == 'easy':
                clean_sentiment = 1
                corrupt_sentiment = 0
            elif temp_data['difficulty'] == 'medium':
                clean_sentiment = 0.8
                corrupt_sentiment = 0.2
            elif temp_data['difficulty'] == 'hard':
                clean_sentiment = 0.6
                corrupt_sentiment = 0.4

            pairs.append(LexicalPair(
                clean_text=clean_text,
                corrupt_text=corrupt_text,
                target_word_clean=pos_word,
                target_word_corrupt=neg_word,
                target_position=target_pos,
                expected_sentiment_flip=True,
                clean_sentiment=clean_sentiment,  # Expected positive score
                corrupt_sentiment=corrupt_sentiment  # Expected negative score
            ))

        return pairs

    def _find_target_position(self, text: str, target_word: str) -> int:
        """Find token position of target word (simplified)"""
        # This is simplified - real implementation needs proper tokenization
        words = text.split()
        try:
            return words.index(target_word) + 1  # +1 for BOS token
        except ValueError:
            return -1


# =============================================================================
# 2. SENTIMENT CLASSIFIER IMPLEMENTATIONS
# =============================================================================

class GPT2SentimentProbe(nn.Module):
    """Linear probe for sentiment classification on GPT-2 representations"""

    def __init__(self, d_model: int = 768, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 2)  # Binary: negative, positive

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            representations: [batch_size, d_model] tensor of GPT-2 representations
        Returns:
            logits: [batch_size, 2] sentiment logits
        """
        x = self.dropout(representations)
        return self.classifier(x)


def create_sentiment_training_data():
    """Create simple training data for the probe"""
    positive_examples = [
        "This movie is amazing and fantastic!",
        "I love this wonderful experience.",
        "Absolutely brilliant and perfect performance.",
        "Outstanding and excellent work here.",
        "This is incredibly good and delightful.",
        "Superb quality and marvelous results.",
        "Fantastic job, truly awesome work.",
        "Excellent service and wonderful staff.",
        "This is absolutely perfect and amazing.",
        "Great experience, highly recommend this."
    ]

    negative_examples = [
        "This movie is terrible and awful.",
        "I hate this horrible experience.",
        "Absolutely dreadful and pathetic performance.",
        "Disappointing and terrible work here.",
        "This is incredibly bad and frustrating.",
        "Poor quality and dreadful results.",
        "Terrible job, truly awful work.",
        "Poor service and rude staff.",
        "This is absolutely horrible and disgusting.",
        "Bad experience, would not recommend this."
    ]

    texts = positive_examples + negative_examples
    labels = [1] * len(positive_examples) + [0] * len(negative_examples)

    return texts, labels


# =============================================================================
# 3. CORE IMPLEMENTATION: LEXICAL PATCHING ANALYZER
# =============================================================================

class LexicalPatchingAnalyzer:
    """Analyze lexical processing in early layers (0-3)"""

    def __init__(self, model_name: str = "gpt2", sentiment_method: str = "probe"):
        self.model = HookedTransformer.from_pretrained(model_name)
        self.model.eval()
        self.device = next(self.model.parameters()).device

        # Choose sentiment classification method
        self.sentiment_method = sentiment_method
        self.sentiment_classifier = self._setup_sentiment_classifier(sentiment_method)

        # Layers to test for Stage 1
        self.target_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        # Generate an integer to use as a seed
        # This can be any integer value you choose


    def _setup_sentiment_classifier(self, method: str = "probe"):
        """Setup sentiment classification with multiple approaches"""

        if method == "probe":
            return self._setup_linear_probe()
        elif method == "external":
            return self._setup_external_model()
        elif method == "simple":
            return self._setup_simple_classifier()
        else:
            raise ValueError(f"Unknown sentiment method: {method}")

    def _setup_linear_probe(self):
        """Setup and train a linear probe on GPT-2 representations"""
        print("Setting up linear probe sentiment classifier...")

        # Initialize probe
        probe = GPT2SentimentProbe(d_model=self.model.cfg.d_model)
        probe.to(self.device)

        # Check if pre-trained probe exists
        probe_path = "gpt2_sentiment_probe.pth"
        if os.path.exists(probe_path):
            print("Loading pre-trained probe...")
            probe.load_state_dict(torch.load(probe_path, map_location=self.device))
            probe.eval()
            return probe

        # Train new probe
        print("Training new sentiment probe...")
        texts, labels = create_sentiment_training_data()

        # Get GPT-2 representations
        representations = []
        with torch.no_grad():
            for text in texts:
                tokens = self.model.to_tokens(text)
                _, cache = self.model.run_with_cache(tokens)
                # Use final token representation from last layer
                final_repr = cache["resid_post", -1][0, -1, :]
                representations.append(final_repr.cpu())

        representations = torch.stack(representations).to(self.device)
        labels = torch.tensor(labels, dtype=torch.long).to(self.device)

        # Train probe
        optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        probe.train()
        for epoch in range(100):
            optimizer.zero_grad()
            logits = probe(representations)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                accuracy = (logits.argmax(dim=1) == labels).float().mean()
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        probe.eval()

        # Save probe
        torch.save(probe.state_dict(), probe_path)
        print("Probe training complete and saved!")

        return probe

    def _setup_external_model(self):
        """Setup external sentiment model (RoBERTa-based)"""
        print("Setting up external sentiment classifier...")

        try:
            # Use a lightweight sentiment model
            classifier = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            print("External sentiment classifier loaded successfully!")
            return classifier

        except Exception as e:
            print(f"Failed to load external model: {e}")
            print("Falling back to simple classifier...")
            return self._setup_simple_classifier()

    def _setup_simple_classifier(self):
        """Setup simple keyword-based classifier"""
        print("Setting up simple keyword-based classifier...")

        positive_words = {
            "amazing", "fantastic", "excellent", "wonderful", "brilliant",
            "outstanding", "superb", "magnificent", "terrific", "awesome",
            "delightful", "marvelous", "incredible", "spectacular", "perfect",
            "great", "good", "love", "best", "beautiful"
        }

        negative_words = {
            "terrible", "awful", "horrible", "disgusting", "pathetic",
            "dreadful", "disappointing", "bad", "poor", "worst",
            "hate", "boring", "annoying", "frustrating", "irritating",
            "unpleasant", "disturbing", "tragic", "sad", "angry"
        }

        return {"positive": positive_words, "negative": negative_words}

    def _get_sentiment(self, text: str) -> float:
        """Get sentiment score for text"""

        if self.sentiment_method == "probe":
            return self._get_sentiment_probe(text)
        elif self.sentiment_method == "external":
            return self._get_sentiment_external(text)
        elif self.sentiment_method == "simple":
            return self._get_sentiment_simple(text)

    def _get_sentiment_probe(self, text: str) -> float:
        """Get sentiment using linear probe"""
        tokens = self.model.to_tokens(text)

        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)
            # Use final token representation from last layer
            final_repr = cache["resid_post", -1][0, -1, :]

            # Get sentiment logits
            logits = self.sentiment_classifier(final_repr.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=-1)

            # Return probability of positive sentiment
            return probabilities[0, 1].item()

    def _get_sentiment_external(self, text: str) -> float:
        """Get sentiment using external model"""
        try:
            results = self.sentiment_classifier(text)

            # Handle different output formats
            if isinstance(results, list) and len(results) > 0:
                scores = results[0] if isinstance(results[0], list) else results

                # Find positive score
                for score_dict in scores:
                    if 'POSITIVE' in score_dict['label'].upper():
                        return score_dict['score']
                    elif 'LABEL_2' == score_dict['label']:  # Some models use LABEL_2 for positive
                        return score_dict['score']

                # Fallback: assume last label is positive
                return scores[-1]['score']

        except Exception as e:
            print(f"External model error: {e}")

        # Fallback to simple method
        return self._get_sentiment_simple(text)

    def _get_sentiment_simple(self, text: str) -> float:
        """Get sentiment using simple keyword matching"""
        text_lower = text.lower()
        words = text_lower.split()

        positive_count = sum(1 for word in words if word in self.sentiment_classifier["positive"])
        negative_count = sum(1 for word in words if word in self.sentiment_classifier["negative"])

        total_sentiment_words = positive_count + negative_count

        if total_sentiment_words == 0:
            return 0.5  # Neutral

        return positive_count / total_sentiment_words

    def _get_sentiment_from_tokens(self, tokens: torch.Tensor, layer) -> float:
        """Get sentiment from pre-tokenized input (used during patching)"""

        if self.sentiment_method == "probe":
            return self._get_sentiment_from_tokens_probe(tokens, layer)
        elif self.sentiment_method == "external":
            return self._get_sentiment_from_tokens_external(tokens, layer)
        elif self.sentiment_method == "simple":
            return self._get_sentiment_from_tokens_simple(tokens, layer)

    def _get_sentiment_from_tokens_probe(self, tokens: torch.Tensor, layer) -> float:
        """Get sentiment from tokens using probe"""
        with torch.no_grad():
            # Run model to get representations
            _, cache = self.model.run_with_cache(tokens)
            final_repr = cache["resid_post", layer][0, -1, :]

            # Get sentiment
            logits = self.sentiment_classifier(final_repr.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=-1)

            return probabilities[0, 1].item()

    def _get_sentiment_from_tokens_external(self, tokens: torch.Tensor) -> float:
        """Get sentiment from tokens using external model"""
        # Convert tokens back to text
        text = self.model.to_string(tokens[0])
        return self._get_sentiment_external(text)

    def _get_sentiment_from_tokens_simple(self, tokens: torch.Tensor) -> float:
        """Get sentiment from tokens using simple method"""
        # Convert tokens back to text
        text = self.model.to_string(tokens[0])
        return self._get_sentiment_simple(text)

    def run_lexical_analysis(self, test_pairs: List[LexicalPair]) -> pd.DataFrame:
        """Run complete lexical analysis on test pairs"""
        results = []

        print(f"Analyzing {len(test_pairs)} lexical pairs across layers {self.target_layers}")

        for pair_idx, pair in tqdm(enumerate(test_pairs)):
            # print(f"Processing pair {pair_idx + 1}/{len(test_pairs)}")

            # Get baseline sentiments
            clean_sentiment = self._get_sentiment(pair.clean_text)
            corrupt_sentiment = self._get_sentiment(pair.corrupt_text)

            # print(f"  Clean: '{pair.clean_text}' -> {clean_sentiment:.3f}")
            # print(f"  Corrupt: '{pair.corrupt_text}' -> {corrupt_sentiment:.3f}")

            # Test each layer
            for layer in self.target_layers:
                # Test full layer patching
                full_effect = self._test_layer_patching(pair, layer, method="full")

                # print('\nLayer: ', layer)
                # print('Full effect: ', full_effect)
                # print('clean_text', pair.clean_text)

                # Test position-specific patching
                pos_effect = self._test_position_patching(pair, layer, pair.target_position)

                # Test off-target position (control)
                control_pos = 0 if pair.target_position != 0 else -1
                control_effect = self._test_position_patching(pair, layer, control_pos)

                results.append({
                    'pair_idx': pair_idx,
                    'layer': layer,
                    'clean_text': pair.clean_text,
                    'target_word': pair.target_word_clean,
                    'target_position': pair.target_position,
                    'baseline_sentiment': clean_sentiment,
                    'corrupt_sentiment': corrupt_sentiment,
                    'full_layer_effect': full_effect,
                    'position_effect': pos_effect,
                    'control_effect': control_effect,
                    'position_specificity': pos_effect - control_effect,
                    'expected_flip': pair.expected_sentiment_flip
                })

        return pd.DataFrame(results)

    def _test_layer_patching(self, pair: LexicalPair, layer: int, method: str = "full") -> float:
        """Test effect of patching entire layer"""
        hook_name = f"blocks.{layer}.hook_resid_post"

        # Get caches
        clean_tokens = self.model.to_tokens(pair.clean_text)
        corrupt_tokens = self.model.to_tokens(pair.corrupt_text)

        with torch.no_grad():
            _, clean_cache = self.model.run_with_cache(clean_tokens)
            _, corrupt_cache = self.model.run_with_cache(corrupt_tokens)

        # Create hook function
        def patch_hook(activation, hook):
            return corrupt_cache[hook_name]

        # Run with patch
        try:
            with torch.no_grad():
                with self.model.hooks([(hook_name, patch_hook)]):
                    patched_sentiment = self._get_sentiment_from_tokens(clean_tokens, layer)

            baseline_sentiment = self._get_sentiment(pair.clean_text)
            effect = abs(patched_sentiment - baseline_sentiment)

            return effect

        except Exception as e:
            print(f"Error in layer {layer} patching: {e}")
            return 0.0

    def _test_position_patching(self, pair: LexicalPair, layer: int, position: int) -> float:
        """Test effect of patching specific position in layer"""
        hook_name = f"blocks.{layer}.hook_resid_post"

        clean_tokens = self.model.to_tokens(pair.clean_text)
        corrupt_tokens = self.model.to_tokens(pair.corrupt_text)

        with torch.no_grad():
            _, clean_cache = self.model.run_with_cache(clean_tokens)
            _, corrupt_cache = self.model.run_with_cache(corrupt_tokens)

        def position_patch_hook(activation, hook):
            patched_activation = activation.clone()
            if position < activation.shape[1]:  # Check bounds
                patched_activation[0, position, :] = corrupt_cache[hook_name][0, position, :]
            return patched_activation

        try:
            with torch.no_grad():
                with self.model.hooks([(hook_name, position_patch_hook)]):
                    patched_sentiment = self._get_sentiment_from_tokens(clean_tokens, layer)

            baseline_sentiment = self._get_sentiment(pair.clean_text)
            effect = abs(patched_sentiment - baseline_sentiment)

            return effect

        except Exception as e:
            print(f"Error in position {position} patching: {e}")
            return 0.0


# =============================================================================
# 4. PERFORMANCE TESTING AND VALIDATION
# =============================================================================

class Stage1PerformanceTester:
    """Test and validate Stage 1 hypotheses"""

    def __init__(self, analyzer: LexicalPatchingAnalyzer):
        self.analyzer = analyzer

    def test_stage1_hypotheses(self, results_df: pd.DataFrame) -> Dict:
        """Test all Stage 1 specific hypotheses"""

        tests = {
            "lexical_sensitivity": self._test_lexical_sensitivity(results_df),
            "early_layer_dominance": self._test_early_layer_dominance(results_df),
            "position_specificity": self._test_position_specificity(results_df),
            "context_independence": self._test_context_independence(results_df)
        }

        return tests

    def _test_lexical_sensitivity(self, df: pd.DataFrame) -> Dict:
        """Test: Early layers should be most sensitive to lexical changes"""

        # Compare target position effects vs control position effects
        target_effects = df.groupby('layer')['position_effect'].mean()
        control_effects = df.groupby('layer')['control_effect'].mean()

        # Expectation: target >> control for early layers
        sensitivity_ratio = control_effects / target_effects

        # Test passes if early layers (0-3) show higher sensitivity ratios
        early_sensitivity = sensitivity_ratio[0:4].mean()
        later_layers = [l for l in sensitivity_ratio.index if l > 3]
        later_sensitivity = sensitivity_ratio[later_layers].mean() if later_layers else 0

        if (early_sensitivity > later_sensitivity):
            hypothesis_supported = 1
        else:
            hypothesis_supported = 0

        return {
            "early_layer_sensitivity": early_sensitivity,
            "later_layer_sensitivity": later_sensitivity,
            "hypothesis_supported": hypothesis_supported,
            "sensitivity_by_layer": sensitivity_ratio.to_dict()
        }

    def _test_early_layer_dominance(self, df: pd.DataFrame) -> Dict:
        """Test: Layers 0-3 should show strongest lexical effects"""

        layer_effects = df.groupby('layer')['position_effect'].mean()

        # Find layer with maximum effect
        max_effect_layer = layer_effects.idxmax()
        max_effect_value = layer_effects.max()

        # Test passes if max effect is in layers 0-3
        if max_effect_layer <= 3:
            hypothesis_supported = 1
        else:
            hypothesis_supported = 0

        return {
            "max_effect_layer": max_effect_layer,
            "max_effect_value": max_effect_value,
            "hypothesis_supported": hypothesis_supported,
            "layer_effects": layer_effects.to_dict()
        }

    def _test_position_specificity(self, df: pd.DataFrame) -> Dict:
        """Test: Effects should be strongest at sentiment word positions"""

        # Compare effects at target positions vs other positions
        position_specificity = df['position_specificity'].mean()

        # Test statistical significance
        try:
            from scipy.stats import ttest_1samp
            t_stat, p_value = ttest_1samp(df['position_specificity'], 0)
        except ImportError:
            # Fallback if scipy not available
            t_stat, p_value = 0, 0.5

        if p_value < 0.05 and position_specificity > 0:
            hypothesis_supported = 1
        else:
            hypothesis_supported = 0

        return {
            "mean_position_specificity": position_specificity,
            "t_statistic": t_stat,
            "p_value": p_value,
            "hypothesis_supported": hypothesis_supported
        }

    def _test_context_independence(self, df: pd.DataFrame) -> Dict:
        """Test: Stage 1 should be relatively context-independent"""

        # Group by different sentence contexts and see if effects are consistent
        # This requires more sophisticated analysis of your test cases

        # Simplified version: check if effects are consistent across pairs
        effect_consistency = df.groupby('layer')['position_effect'].std()

        # Lower standard deviation indicates more consistent (context-independent) effects
        early_consistency = effect_consistency[0:4].mean()
        later_consistency = effect_consistency[4:].mean()

        if early_consistency < later_consistency:
            hypothesis_supported = 1
        else:
            hypothesis_supported = 0

        return {
            "early_layer_consistency": early_consistency,
            "effect_std_by_layer": effect_consistency.to_dict(),
            "hypothesis_supported": hypothesis_supported  # Arbitrary threshold
        }


# =============================================================================
# 5. VISUALIZATION AND ANALYSIS
# =============================================================================

class Stage1Visualizer:
    """Create visualizations for Stage 1 analysis"""

    @staticmethod
    def plot_layer_effects(results_df: pd.DataFrame):
        """Plot patching effects by layer"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Layer-wise effects
        layer_effects = results_df.groupby('layer')['position_effect'].mean()
        axes[0, 0].bar(layer_effects.index, layer_effects.values)
        axes[0, 0].set_title('Average Position Effect by Layer')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Effect Size')

        # Position specificity by layer
        pos_spec = results_df.groupby('layer')['position_specificity'].mean()
        axes[0, 1].bar(pos_spec.index, pos_spec.values)
        axes[0, 1].set_title('Position Specificity by Layer')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Specificity (Target - Control)')

        # Full layer effects
        full_effects = results_df.groupby('layer')['full_layer_effect'].mean()
        axes[1, 0].bar(full_effects.index, full_effects.values)
        axes[1, 0].set_title('Full Layer Effect by Layer')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Effect Size')

        # Distribution of effects
        results_df.boxplot(column='position_effect', by='layer', ax=axes[1, 1])
        axes[1, 1].set_title('Distribution of Position Effects')

        plt.tight_layout()
        # plt.show()

        plt.savefig(f"plot_{my_seed}.png")

    @staticmethod
    def plot_word_sensitivity_heatmap(results_df: pd.DataFrame):
        """Create heatmap of word-specific effects"""
        results_df.to_csv('test.csv', index=False)
        # Pivot to create word x layer matrix
        heatmap_data = results_df.pivot_table(
            values='position_effect',
            index='target_word',
            columns='layer',
            aggfunc='mean'
        )
        #Seed used 16
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis')
        plt.title('Sentiment Word Sensitivity by Layer')
        plt.xlabel('Layer')
        plt.ylabel('Target Sentiment Word')
        # plt.show()
        plt.savefig(f"hitmap_{my_seed}.png")

# =============================================================================
# 6. COMPLETE TESTING PIPELINE
# =============================================================================

def run_stage1_complete_analysis(sentiment_method: str = "probe"):
    """Complete pipeline for Stage 1 analysis"""

    print("🚀 Starting Stage 1: Lexical Detection Analysis")

    # Step 1: Generate test data
    print("\n📊 Step 1: Generating lexical test pairs...")
    test_suite = LexicalTestSuite()
    lexical_pairs = test_suite.generate_lexical_pairs(num_pairs=20)  # Start small
    print(f"Generated {len(lexical_pairs)} lexical test pairs")

    # Step 2: Initialize analyzer
    print("\n🔧 Step 2: Initializing analyzer...")
    analyzer = LexicalPatchingAnalyzer(sentiment_method=sentiment_method)

    # Step 3: Run analysis
    print("\n⚡ Step 3: Running patching experiments...")
    results_df = analyzer.run_lexical_analysis(lexical_pairs)

    # Step 4: Test hypotheses
    print("\n🧪 Step 4: Testing Stage 1 hypotheses...")
    tester = Stage1PerformanceTester(analyzer)
    hypothesis_results = tester.test_stage1_hypotheses(results_df)

    # Step 5: Print results
    print("\n📋 Step 5: Results Summary")
    print("=" * 50)

    for test_name, test_results in hypothesis_results.items():
        status = "✅ SUPPORTED" if test_results.get('hypothesis_supported', False) else "❌ NOT SUPPORTED"
        print(f"{test_name.upper()}: {status}")

    # Step 6: Visualize
    print("\n📈 Step 6: Generating visualizations...")
    try:
        Stage1Visualizer.plot_layer_effects(results_df)
        Stage1Visualizer.plot_word_sensitivity_heatmap(results_df)
    except Exception as e:
        print(f"Visualization error: {e}")

    # Step 7: Save results
    results_df.to_csv('stage1_lexical_results.csv', index=False)

    with open('stage1_hypothesis_tests.json', 'w') as f:
        # Convert numpy types to Python native types for JSON serialization
        serializable_results = {}
        for key, value in hypothesis_results.items():
            serializable_results[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, (np.int64, np.int32)):
                    serializable_results[key][subkey] = int(subvalue)
                elif isinstance(subvalue, (np.float64, np.float32)):
                    serializable_results[key][subkey] = float(subvalue)
                else:
                    serializable_results[key][subkey] = subvalue

        json.dump(serializable_results, f, indent=2)

    print("\n🎉 Stage 1 analysis complete!")
    print("Results saved to: stage1_lexical_results.csv")
    print("Hypothesis tests saved to: stage1_hypothesis_tests.json")

    return results_df, hypothesis_results


# =============================================================================
# 7. TESTING FUNCTIONS
# =============================================================================

def test_sentiment_classifiers():
    """Test all three sentiment classification approaches"""

    test_texts = [
        "This movie is absolutely amazing and wonderful!",
        "I hate this terrible and awful experience.",
        "The weather is okay today.",
        "Outstanding performance, truly excellent work!",
        "Disappointing results, very poor quality."
    ]

    for method in ["probe", "simple"]:  # Skip external for now due to potential dependency issues
        print(f"\n--- Testing {method.upper()} method ---")

        try:
            analyzer = LexicalPatchingAnalyzer(sentiment_method=method)

            for text in test_texts:
                sentiment = analyzer._get_sentiment(text)
                print(f"'{text}' -> {sentiment:.3f}")

        except Exception as e:
            print(f"Error with {method}: {e}")


def quick_test():
    """Quick test with minimal data for debugging"""
    print("🔍 Running quick test...")

    # Create a few test pairs manually
    pairs = [
        LexicalPair(
            clean_text="The movie was amazing",
            corrupt_text="The movie was terrible",
            target_word_clean="amazing",
            target_word_corrupt="terrible",
            target_position=4,
            expected_sentiment_flip=True,
            clean_sentiment=0.8,
            corrupt_sentiment=0.2
        ),
        LexicalPair(
            clean_text="This book is excellent",
            corrupt_text="This book is awful",
            target_word_clean="excellent",
            target_word_corrupt="awful",
            target_position=3,
            expected_sentiment_flip=True,
            clean_sentiment=0.8,
            corrupt_sentiment=0.2
        )
    ]

    # Test with simple classifier first
    analyzer = LexicalPatchingAnalyzer(sentiment_method="probe")

    print("Testing sentiment classification...")
    for pair in pairs:
        clean_sent = analyzer._get_sentiment(pair.clean_text)
        corrupt_sent = analyzer._get_sentiment(pair.corrupt_text)
        print(f"Clean: '{pair.clean_text}' -> {clean_sent:.3f}")
        print(f"Corrupt: '{pair.corrupt_text}' -> {corrupt_sent:.3f}")

    print("Testing patching...")
    for i in range(20):
        results = analyzer.run_lexical_analysis(pairs[:i])  # Just test one pair
        print(f"Results shape: {results.shape}")
        print(results.head())

    return results


# =============================================================================
# 8. USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Run Stage 1 Lexical Detection Analysis')
    # parser.add_argument('--method', choices=['probe', 'external', 'simple'],
    #                     default='simple', help='Sentiment classification method')
    # parser.add_argument('--quick', action='store_true',
    #                     help='Run quick test with minimal data')
    # parser.add_argument('--test-classifiers', action='store_true',
    #                     help='Test all sentiment classifiers')

    # args = parser.parse_args()

    # if args.test_classifiers:
    #     test_sentiment_classifiers()
    # elif args.quick:
    #     quick_test()
    # else:

    # Run the complete Stage 1 analysis
    results, hypothesis_tests = run_stage1_complete_analysis(sentiment_method='probe')

    # Print key findings
    print("\n🔍 Key Findings:")
    if not results.empty:
        best_layer = results.groupby('layer')['position_effect'].mean().idxmax()
        avg_specificity = results['position_specificity'].mean()
        significant_pairs = (results['position_effect'] > 0.05).sum()

        print(f"- Best performing layer: {best_layer}")
        print(f"- Average position specificity: {avg_specificity:.3f}")
        print(f"- Number of pairs showing expected effects: {significant_pairs}")
    else:
        print("- No results generated")

# =============================================================================
# 9. ADDITIONAL UTILITY FUNCTIONS
# =============================================================================

# def load_and_analyze_results(results_file: str = 'stage1_lexical_results.csv'):
#     """Load and re-analyze saved results"""
#     try:
#         results_df = pd.read_csv(results_file)

#         print(f"Loaded results with {len(results_df)} rows")
#         print("\nDataset summary:")
#         print(results_df.describe())

#         # Re-run visualizations
#         print("\nGenerating visualizations from saved data...")
#         Stage1Visualizer.plot_layer_effects(results_df)
#         Stage1Visualizer.plot_word_sensitivity_heatmap(results_df)

#         return results_df

#     except FileNotFoundError:
#         print(f"Results file {results_file} not found. Run analysis first.")
#         return None


# def compare_sentiment_methods():
#     """Compare different sentiment classification methods"""
#     print("🔄 Comparing sentiment classification methods...")

#     test_texts = [
#         "This movie is absolutely amazing!",
#         "I hate this terrible experience.",
#         "The service was okay.",
#         "Brilliant and outstanding work!",
#         "Poor and disappointing quality."
#     ]

#     methods = ["simple", "probe"]
#     results = {method: [] for method in methods}

#     for method in methods:
#         print(f"\nTesting {method} method...")
#         try:
#             analyzer = LexicalPatchingAnalyzer(sentiment_method=method)

#             for text in test_texts:
#                 sentiment = analyzer._get_sentiment(text)
#                 results[method].append(sentiment)
#                 print(f"  '{text}' -> {sentiment:.3f}")

#         except Exception as e:
#             print(f"  Error with {method}: {e}")
#             results[method] = [0.5] * len(test_texts)

#     # Compare results
#     print("\n📊 Comparison Summary:")
#     comparison_df = pd.DataFrame(results, index=test_texts)
#     print(comparison_df)

#     # Calculate correlation between methods
#     if len(results["simple"]) == len(results["probe"]):
#         correlation = np.corrcoef(results["simple"], results["probe"])[0, 1]
#         print(f"\nCorrelation between methods: {correlation:.3f}")

#     return comparison_df


# def debug_patching_step_by_step():
#     """Debug the patching process step by step"""
#     print("🐛 Debug: Step-by-step patching analysis")

#     # Simple test case
#     clean_text = "The movie was amazing"
#     corrupt_text = "The movie was terrible"

#     analyzer = LexicalPatchingAnalyzer(sentiment_method="simple")

#     print(f"Clean text: '{clean_text}'")
#     print(f"Corrupt text: '{corrupt_text}'")

#     # Test tokenization
#     clean_tokens = analyzer.model.to_tokens(clean_text)
#     corrupt_tokens = analyzer.model.to_tokens(corrupt_text)

#     print(f"Clean tokens: {clean_tokens}")
#     print(f"Corrupt tokens: {corrupt_tokens}")
#     print(f"Clean token strings: {[analyzer.model.to_string(t) for t in clean_tokens[0]]}")
#     print(f"Corrupt token strings: {[analyzer.model.to_string(t) for t in corrupt_tokens[0]]}")

#     # Test sentiment classification
#     clean_sentiment = analyzer._get_sentiment(clean_text)
#     corrupt_sentiment = analyzer._get_sentiment(corrupt_text)

#     print(f"Clean sentiment: {clean_sentiment:.3f}")
#     print(f"Corrupt sentiment: {corrupt_sentiment:.3f}")

#     # Test caching
#     with torch.no_grad():
#         _, clean_cache = analyzer.model.run_with_cache(clean_tokens)
#         _, corrupt_cache = analyzer.model.run_with_cache(corrupt_tokens)

#     print(f"Cache keys: {list(clean_cache.keys())[:5]}...")  # Show first 5 keys

#     # Test single layer patching
#     layer = 1
#     hook_name = f"blocks.{layer}.hook_resid_post"

#     print(f"Testing layer {layer} patching...")
#     print(f"Hook name: {hook_name}")
#     print(f"Clean cache shape: {clean_cache[hook_name].shape}")
#     print(f"Corrupt cache shape: {corrupt_cache[hook_name].shape}")

#     # Simple position patch test
#     def test_patch_hook(activation, hook):
#         patched = activation.clone()
#         patched[0, -1, :] = corrupt_cache[hook_name][0, -1, :]  # Patch last position
#         return patched

#     try:
#         with torch.no_grad():
#             with analyzer.model.hooks([(hook_name, test_patch_hook)]):
#                 patched_sentiment = analyzer._get_sentiment_from_tokens(clean_tokens)

#         print(f"Patched sentiment: {patched_sentiment:.3f}")
#         print(f"Effect size: {abs(patched_sentiment - clean_sentiment):.3f}")

#     except Exception as e:
#         print(f"Patching error: {e}")
#         import traceback
#         traceback.print_exc()


# # =============================================================================
# # 10. FINAL COMPLETE IMPLEMENTATION
# # =============================================================================

# def main():
#     """Main function with command line interface"""
#     print("🎯 Stage 1: Lexical Detection Analysis")
#     print("=" * 50)

#     print("Available commands:")
#     print("1. Run full analysis")
#     print("2. Quick test")
#     print("3. Test classifiers")
#     print("4. Compare methods")
#     print("5. Debug patching")
#     print("6. Load and analyze results")

#     try:
#         #choice = input("\nSelect option (1-6): ").strip()

#         run_stage1_complete_analysis(sentiment_method='probe')

#         if choice == "1":
#             method = input("Select method (simple/probe/external) [simple]: ").strip() or "simple"
#             run_stage1_complete_analysis(sentiment_method=method)

#         elif choice == "2":
#             quick_test()

#         elif choice == "3":
#             test_sentiment_classifiers()

#         elif choice == "4":
#             compare_sentiment_methods()

#         elif choice == "5":
#             debug_patching_step_by_step()

#         elif choice == "6":
#             load_and_analyze_results()

#         else:
#             print("Invalid choice. Running quick test...")
#             quick_test()

#     except KeyboardInterrupt:
#         print("\n👋 Analysis interrupted by user")
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         print("Running debug mode...")
#         debug_patching_step_by_step()


# # Run main if script is executed directly
# if __name__ == "__main__":
#     main()