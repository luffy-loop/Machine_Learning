# Machine Learning & Generative AI

A hands-on learning repository for machine learning coursework and foundational generative AI implementations in Python. The notebooks preserve academic practice, while the standalone scripts explore the building blocks of transformer models.

## Repository structure

| Directory | Focus |
| --- | --- |
| [ML2-1](ML2-1/) | Machine learning lab notebooks and supporting datasets |
| [Generative_AI](Generative_AI/) | PyTorch attention implementations and Gemini API experiments |

## Generative AI learning path

| Topic | Implementation | Status |
| --- | --- | --- |
| First LLM API call | [first_llm_call.py](Generative_AI/first_llm_call.py) | Implemented |
| Scaled dot-product attention | [scaled_dot_product_attention.py](Generative_AI/scaled_dot_product_attention.py) | Implemented |
| Multi-head attention | [multi_head_attention.py](Generative_AI/multi_head_attention.py) | Implemented |
| Causal masking | [causal_masking.py](Generative_AI/causal_masking.py) | Implemented |
| Positional encoding, transformer blocks and training | Future exercises | Planned |

The attention examples expose query, key and value projections, scaled scores, softmax weights, head splitting and merging, and masking. The causal masking example checks that attention weights for future tokens are zero.

## Machine learning coursework

The [ML2-1](ML2-1/) directory contains lab notebooks for Labs 1–5 and 7, a skill exercise, and student placement and Titanic datasets. Open the notebooks to inspect the individual exercises, code and recorded outputs. The original coursework is retained as a learning record.

## Getting started

Clone the repository and create a Python virtual environment:

```bash
git clone https://github.com/luffy-loop/Machine_Learning.git
cd Machine_Learning
python -m venv .venv
```

Activate on Windows PowerShell with `.\.venv\Scripts\Activate.ps1` or on Linux/macOS with `source .venv/bin/activate`.

Install the dependencies for the GenAI examples:

```bash
python -m pip install torch google-genai python-dotenv
```

Run the attention examples without an API key:

```bash
python Generative_AI/scaled_dot_product_attention.py
python Generative_AI/multi_head_attention.py
python Generative_AI/causal_masking.py
```

For the Gemini example, create a local `.env` containing `GEMINI_API_KEY=your_key_here`, then run `python Generative_AI/first_llm_call.py`. A valid key, network access and access to the model configured in the script are required. Never commit credentials.

For notebooks, install Jupyter (`python -m pip install jupyterlab`) and any additional libraries imported by the selected notebook, then launch `jupyter lab` from the repository root.

## Related applied project

[FraudLens](https://github.com/luffy-loop/FraudLens) demonstrates an applied machine learning workflow for fraud detection in a separate repository.

## Scope and limitations

This is an evolving educational repository, not a production ML framework or a trained language model. Examples are intended to illustrate concepts; automated regression testing and pinned reproducible environments are future improvements. The notebooks may require additional dependencies or local path adjustments.

Maintained by [luffy-loop](https://github.com/luffy-loop).
