# Mechanistic Interpretability of GPT-2: Lexical and Contextual Layers in Sentiment Analysis

This repository contains the code and data for our systematic investigation of how GPT-2 processes sentiment through hierarchical stages using mechanistic interpretability techniques.

## Overview

We present a causal analysis of GPT-2's sentiment processing mechanisms using activation patching across all 12 layers. Our study reveals a two-stage architecture:

1. **Stage 1 (Lexical Detection)**: Early layers (0-3) encode token-level sentiment features with high position specificity
2. **Stage 2 (Contextual Integration)**: Late layers (8-11) handle complex contextual modifications including negation, sarcasm, and intensification

## Key Findings

- **Two-stage processing validated**: Lexical detection peaks in Layer 1; contextual integration concentrates in Layers 8-11
- **Three hypotheses systematically tested**:
  - ❌ Middle Layer Concentration: Falsified (57% late-peaking, 43% early-peaking, 0% mid-peaking)
  - ❌ Phenomenon Specificity: Falsified (87% of phenomena share top-3 layers [11,10,9])
  - ❌ Distributed Processing: Falsified (6.7-fold monotonic increase from Layer 0→11)
- **Phenomenon-agnostic hub**: Most contextual phenomena converge on late-layer processing

⚠️ **Important**: These findings are heuristic results specific to our dataset and may not generalize to other conditions.

## Repository Structure

```
├── activation_patching/
│   ├── phase_1/                    # Lexical detection experiments
│   │   ├── Lexical_Detection.py
│   │   └── lexical_analysis.py
│   └── phase_2/                    # Contextual integration experiments
│       ├── Context_Integration.py
│       └── contextual_analysis.py
├── data/
│   ├── lexical_2000_pairs.csv      # Lexical detection test cases
│   └── sentiment_2000_pairs.csv    # Contextual integration test cases (7,998 pairs)
├── figures/                         # Generated visualizations
├── results/                         # Experimental outputs
└── README.md
```

## Dataset

### Lexical Detection Dataset (2,000 pairs)
Controlled pairs testing context-independent sentiment word recognition:
- Example: "The movie was wonderful" vs "The movie was terrible"

### Contextual Integration Dataset (8,000 pairs)
14 contextual phenomena across diverse modification types:

| Context Type | Code | Examples | Count |
|-------------|------|----------|-------|
| Strong Positive | C1 | "incredible" → "abysmal" | 500 |
| Medium Intensity | C2 | "fine" → "bad" | 500 |
| Intensified Swap | C3 | "utterly wonderful" → "utterly awful" | 500 |
| Comparative Context | C4 | "better than expected" → "worse than expected" | 1,000 |
| Simple Negation | C5 | "nice" → "not nice" | 666 |
| Intensified Negation | C6 | "was outstanding" → "wasn't outstanding" | 666 |
| Complex Double Negation | C7 | "wasn't bad" → "wasn't good" | 666 |
| Domain Context | C8 | "horror movie: haunting" → "comedy: haunting" | 500 |
| Sarcasm | C9 | Ironic context modifications | 500 |
| Conditional vs Actual | C10 | "would have been" → "was" | 500 |
| Intensity Variation | C11 | "incredibly" → "a bit" | 500 |
| Multiple Intensifiers | C12 | "utterly very" → "just" | 500 |
| Intensity Flip | C13 | "extremely" → "only slightly" | 500 |
| Scale Variation | C14 | Different sentiment scale positions | 500 |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mechanistic-interpretability-gpt2
cd mechanistic-interpretability-gpt2

# Create conda environment
conda create -n mechanistic_interpretability python=3.9
conda activate mechanistic_interpretability

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.9+
- PyTorch 2.0+
- TransformerLens
- pandas
- numpy
- plotly
- tqdm

## Usage

### Running Lexical Detection Analysis

```bash
python activation_patching/phase_1/Lexical_Detection.py
```

This generates:
- Layer sensitivity scores across all 12 layers
- Position specificity analysis
- Context independence measurements

### Running Contextual Integration Analysis

```bash
python activation_patching/phase_2/Context_Integration.py
```

This generates:
- Peak layer distribution across 14 context types
- Layer importance gradients
- Phenomenon-specific activation patterns

### Generating Visualizations

```python
# Figure 4: Peak Layer Distribution
python generate_figures.py --figure peak_distribution

# Figure: Layer Importance Gradient
python generate_figures.py --figure importance_gradient
```

## Key Results

### Lexical Detection (Stage 1)
- **Peak Layer**: Layer 1 (early)
- **Position Specificity**: 0.147 mean score (p < 0.001)
- **Context Independence**: Low variability in early layers (0-3)

### Contextual Integration (Stage 2)
- **Most Common Peak**: Layer 11 (8/14 phenomena)
- **Top 5 Layers**: 11 (5,537), 10 (5,446), 9 (5,281), 8 (5,145), 7 (4,989)
- **Layer Importance Distribution**:
  - Early (0-3): 15%
  - Mid (4-7): 39%
  - Late (8-11): 46%

### Hypothesis Testing Results

| Hypothesis | Prediction | Result | Evidence |
|-----------|-----------|--------|----------|
| Middle Layer Concentration | Peak in layers 4-8 | ❌ Falsified | 0% peak in mid-layers |
| Phenomenon Specificity | Distinct patterns per phenomenon | ❌ Falsified | 87% share [11,10,9] |
| Distributed Processing | Uniform distribution | ❌ Falsified | 6.7x increase L0→L11 |

## Methodology

1. **Activation Patching**: Replace activations from source sentence with target sentence at each layer
2. **Causal Measurement**: Quantify change in sentiment classification probability
3. **Controlled Testing**: Minimal pairs differing only in target modification
4. **Statistical Validation**: 2,000-8,000 test cases per analysis

## Citation

```bibtex
@inproceedings{hatua2025mechanistic,
  title={Mechanistic Interpretability of GPT-2: Lexical and Contextual Layers in Sentiment Analysis},
  author={Hatua, Amartya},
  booktitle={NeurIPS 2025 Workshop on Efficient Reasoning},
  year={2025}
}
```

## Limitations

- **Dataset-specific**: Results may not generalize beyond our test cases
- **Model-specific**: Tested only on GPT-2 (117M parameters)
- **Task-specific**: Limited to sentiment analysis
- **Layer-level analysis**: Does not examine individual attention heads or neurons

## Future Work

- Validate across diverse architectures (BERT, RoBERTa, larger GPT models)
- Fine-grained circuit-level analysis of attention heads and MLP blocks
- Extended datasets with implicit sentiment and multi-sentence reasoning
- Training dynamics analysis at different checkpoints

## License

MIT License - see LICENSE file for details

## Contact

For questions or collaborations, please open an issue or contact [amartyahatua@gmail.com]

## Acknowledgments

- TransformerLens library for model access and intervention tools
- Anthropic and OpenAI for mechanistic interpretability frameworks
- NeurIPS 2025 reviewers for valuable feedback

---

**Disclaimer**: These findings represent heuristic results specific to our experimental conditions and should not be interpreted as definitive claims about transformer architecture in general.
