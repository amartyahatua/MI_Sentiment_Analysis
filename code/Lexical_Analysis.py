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
# =============================================================================
# 1. DATA STRUCTURES FOR LEXICAL TESTING
# =============================================================================

@dataclass
class LexicalPair:
    """
    Represents a clean vs corrupted lexical test case for sentiment analysis.

    Attributes:
        clean_text (str): Original input sentence containing a sentiment word
        corrupt_text (str): Modified version with swapped sentiment word
        target_word_clean (str): The positive sentiment word in clean text
        target_word_corrupt (str): The negative sentiment word in corrupt text
        target_position (int): Token index of the target word in the sequence
        expected_sentiment_flip (bool): Should sentiment flip when patched?
        clean_sentiment (float): Expected sentiment score for clean text
        corrupt_sentiment (float): Expected sentiment score for corrupt text
    """

    clean_text: str
    corrupt_text: str
    target_word_clean: str  # The sentiment word in clean text
    target_word_corrupt: str  # The sentiment word in corrupt text
    target_position: int  # Token position of target word
    expected_sentiment_flip: bool  # Should sentiment flip when patched?
    clean_sentiment: float  # Expected sentiment score for clean
    corrupt_sentiment: float  # Expected sentiment score for corrupt


class LexicalTestSuite:
    """
    Generates and manages lexical test pairs for probing sentiment at word level.

    Contains predefined lists of:
    - positive_words: 15 positive sentiment adjectives
    - negative_words: 15 negative sentiment adjectives
    - neutral_contexts: 8 sentence templates
    - intensifiers: Common amplifying words
    """

    def __init__(self):
        """
                Initialize test suite with lists of positive and negative words.

                Args:
                    positive_words (list): List of positive sentiment words.
                    negative_words (list): List of negative sentiment words.
        """

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

    def generate_lexical_pairs(self) -> List[LexicalPair]:
        """
                Generate lexical test pairs from CSV data for sentiment analysis probing.

                Reads from '../data/sentiment_2000_pairs.csv' and processes first 200 pairs.
                Creates clean/corrupt text pairs by swapping positive/negative sentiment words.

                Returns:
                    List[LexicalPair]: Test pairs with sentiment words swapped, including:
                    - Difficulty-based sentiment scores (easy: 1.0/0.0, medium: 0.8/0.2, hard: 0.6/0.4)
                    - Target word positions extracted from CSV
                    - Expected sentiment flip behavior
        """
        pairs = []
        data = pd.read_csv('../data/sentiment_2000_pairs.csv')
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
    """
    Linear probe for sentiment classification on GPT-2 representations.

    Architecture:
    - Input: GPT-2 hidden representations (768-dimensional)
    - Dropout layer (0.1 default)
    - Linear classifier to 2 classes (negative, positive)

    forward(representations: torch.Tensor) -> torch.Tensor:
        Forward pass returning sentiment logits [batch_size, 2]
    """

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
    """
    Create simple training data for the sentiment probe.

    Returns:
        texts (list): 20 example sentences (10 positive, 10 negative)
        labels (list): Binary labels (1=positive, 0=negative)

    Used to train the linear probe on GPT-2 representations for sentiment classification.
    """
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
    """
    Main analyzer for lexical processing in early transformer layers (0-11).

    Supports three sentiment classification methods:
    1. "probe": Linear probe trained on GPT-2 representations
    2. "external": RoBERTa-based sentiment classifier
    3. "simple": Keyword-based classification
    """

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
        """
        Initialize sentiment classification system.

        Args:
            method: Classification approach ("probe", "external", "simple")

        Returns:
            Configured sentiment classifier based on chosen method

        - "probe": Trains/loads linear probe on GPT-2 representations
        - "external": Uses cardiffnlp/twitter-roberta-base-sentiment-latest
        - "simple": Keyword matching with predefined word lists
        """

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
        """
        Get sentiment score for input text using configured classifier.

        Args:
            text: Input sentence to analyze

        Returns:
            float: Sentiment probability (0.0 = negative, 1.0 = positive)

        Delegates to appropriate method based on self.sentiment_method
        """

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
        """
        Run complete lexical analysis using activation patching.

        Args:
            test_pairs: List of LexicalPair objects to analyze

        Returns:
            pd.DataFrame with columns:
            - pair_idx: Test pair identifier
            - layer: Transformer layer (0-11)
            - clean_text: Original sentence
            - target_word: Sentiment word being tested
            - target_position: Token position of sentiment word
            - baseline_sentiment: Sentiment score for clean text
            - corrupt_sentiment: Sentiment score for corrupt text
            - full_layer_effect: Effect of patching entire layer
            - position_effect: Effect of patching specific position
            - control_effect: Effect of patching control position
            - position_specificity: position_effect - control_effect
            - expected_flip: Whether sentiment should flip

        Process:
        1. For each test pair and each layer (0-11):
        2. Test full layer patching (replace entire layer activations)
        3. Test position-specific patching (replace single token position)
        4. Test control position patching (off-target position)
        5. Calculate effect sizes as absolute differences in sentiment
        """
        results = []

        print(f"Analyzing {len(test_pairs)} lexical pairs across layers {self.target_layers}")

        for pair_idx, pair in tqdm(enumerate(test_pairs)):
            # print(f"Processing pair {pair_idx + 1}/{len(test_pairs)}")

            # Get baseline sentiments
            clean_sentiment = self._get_sentiment(pair.clean_text)
            corrupt_sentiment = self._get_sentiment(pair.corrupt_text)

            # Test each layer
            for layer in self.target_layers:
                # Test full layer patching
                full_effect = self._test_layer_patching(pair, layer, method="full")

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
        """
        Test effect of patching entire layer with corrupt activations.

        Args:
            pair: Test case with clean/corrupt texts
            layer: Target layer to patch
            method: Patching method ("full" for entire layer)

        Returns:
            float: Effect size (absolute change in sentiment score)

        Process:
        1. Get clean and corrupt activations from specified layer
        2. Run clean input with corrupt activations patched in
        3. Measure sentiment change compared to baseline
        """
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
        """
        Test effect of patching specific token position in layer.

        Args:
            pair: Test case with clean/corrupt texts
            layer: Target layer to patch
            position: Token position to patch

        Returns:
            float: Effect size (absolute change in sentiment score)

        Process:
        1. Get activations for clean and corrupt inputs
        2. Replace activation at specified position with corrupt version
        3. Measure resulting sentiment change
        4. Used to test position-specificity of lexical effects
        """
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
    """
    Test and validate Stage 1 hypotheses about lexical processing.

    Tests four key hypotheses:
    1. Lexical sensitivity: Early layers more sensitive to lexical changes
    2. Early layer dominance: Layers 0-3 show strongest effects
    3. Position specificity: Effects strongest at sentiment word positions
    4. Context independence: Stage 1 processing is context-independent
    """

    def __init__(self, analyzer: LexicalPatchingAnalyzer):
        self.analyzer = analyzer

    def test_stage1_hypotheses(self, results_df: pd.DataFrame) -> Dict:
        """
        Run all Stage 1 hypothesis tests on experimental results.

        Args:
            results_df: DataFrame from run_lexical_analysis()

        Returns:
            Dict with test results for each hypothesis:
            - "lexical_sensitivity": Early vs late layer sensitivity comparison
            - "early_layer_dominance": Whether layers 0-3 show max effects
            - "position_specificity": Statistical test of position effects
            - "context_independence": Consistency of effects across contexts

        Each test returns hypothesis_supported (0/1) plus detailed metrics.
        """

        tests = {
            "lexical_sensitivity": self._test_lexical_sensitivity(results_df),
            "early_layer_dominance": self._test_early_layer_dominance(results_df),
            "position_specificity": self._test_position_specificity(results_df),
            "context_independence": self._test_context_independence(results_df)
        }

        return tests

    def _test_lexical_sensitivity(self, df: pd.DataFrame) -> Dict:
        """
        Test: Early layers should be most sensitive to lexical changes.

        Compares target position effects vs control position effects across layers.
        Expects higher sensitivity ratios (control/target) in early layers.

        Returns:
            Dict with early/late sensitivity comparison and support verdict.
        """

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
        """
        Test: Layers 0-3 should show strongest lexical effects.

        Finds layer with maximum average position effect.
        Passes if maximum effect occurs in layers 0-3.

        Returns:
            Dict with max effect layer, value, and support verdict.
        """

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
        """
        Test: Effects should be strongest at sentiment word positions.

        Uses t-test to check if position_specificity significantly > 0.
        position_specificity = position_effect - control_effect

        Returns:
            Dict with mean specificity, t-statistic, p-value, and support verdict.
        """

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
        """
        Test: Stage 1 should be relatively context-independent.

        Measures consistency of effects across different sentence contexts.
        Lower standard deviation indicates more context-independent processing.

        Returns:
            Dict comparing early vs late layer consistency and support verdict.
        """
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
# 6. COMPLETE TESTING PIPELINE
# =============================================================================

def run_stage1_complete_analysis(sentiment_method: str = "probe"):
    """Complete pipeline for Stage 1 analysis"""

    print("Starting Stage 1: Lexical Detection Analysis")

    # Step 1: Generate test data
    print("\n📊 Step 1: Generating lexical test pairs...")
    test_suite = LexicalTestSuite()
    lexical_pairs = test_suite.generate_lexical_pairs()  # Start small
    print(f"Generated {len(lexical_pairs)} lexical test pairs")

    # Step 2: Initialize analyzer
    print("Step 2: Initializing analyzer...")
    analyzer = LexicalPatchingAnalyzer(sentiment_method=sentiment_method)

    # Step 3: Run analysis
    print("Step 3: Running patching experiments...")
    results_df = analyzer.run_lexical_analysis(lexical_pairs)

    # Step 4: Test hypotheses
    print("Step 4: Testing Stage 1 hypotheses...")
    tester = Stage1PerformanceTester(analyzer)
    hypothesis_results = tester.test_stage1_hypotheses(results_df)

    # Step 5: Print results
    print("Step 5: Results Summary")
    print("=" * 50)

    for test_name, test_results in hypothesis_results.items():
        status = "SUPPORTED" if test_results.get('hypothesis_supported', False) else "NOT SUPPORTED"
        print(f"{test_name.upper()}: {status}")

    # Step 7: Save results
    results_df.to_csv('../result/Lexical_Analysis_Results.csv', index=False)
    print("Stage 1 analysis complete!")
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
    print("Running quick test...")

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
    # Run the complete Stage 1 analysis
    results, hypothesis_tests = run_stage1_complete_analysis(sentiment_method='probe')

    # Print key findings
    print("Key Findings:")
    if not results.empty:
        best_layer = results.groupby('layer')['position_effect'].mean().idxmax()
        avg_specificity = results['position_specificity'].mean()
        significant_pairs = (results['position_effect'] > 0.05).sum()

        print(f"- Best performing layer: {best_layer}")
        print(f"- Average position specificity: {avg_specificity:.3f}")
        print(f"- Number of pairs showing expected effects: {significant_pairs}")
    else:
        print("- No results generated")