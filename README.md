# SHAP Explainability for Semantic Retrieval

This project implements SHAP (SHapley Additive exPlanations) analysis for explaining semantic similarity rankings in document retrieval systems. It helps understand why a transformer-based semantic model ranks certain documents as highly relevant to a given query.

## Features

- **Multi-Query Analysis**: Process multiple research queries with their associated PMC article sets
- **Database Integration**: Fetch article content from SQL Server database using PMC IDs
- **SHAP Explanations**: Generate token-level explanations for semantic similarity scores
- **Stop Word Filtering**: Remove common English stop words for cleaner analysis
- **Multiple Output Formats**: 
  - CSV files with token contributions
  - Bar chart visualizations
  - JSON exports with full explanation data
- **Manual Analysis**: Run SHAP on individual text documents

## Project Structure

```
├── main.py                    # Main pipeline for multi-query SHAP analysis
├── manual_shapley.py          # Single document SHAP analysis
├── manual_shapley_data.txt    # Sample text for manual analysis
├── infrastructure/
│   └── article_repository.py  # Database connection and article fetching
├── data/
│   ├── query_1_pmc_ids.txt   # PMC IDs for query 1
│   ├── query_2_pmc_ids.txt   # PMC IDs for query 2
│   └── query_3_pmc_ids.txt   # PMC IDs for query 3
├── shap_outputs/
│   ├── query_1/              # SHAP results for query 1
│   ├── query_2/              # SHAP results for query 2
│   └── query_3/              # SHAP results for query 3
└── manual_shap_outputs/       # Manual analysis results
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mrk007Git/dhs-dissertation-shapley.git
cd dhs-dissertation-shapley
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env`:
```
DB_SERVER=your-server.database.windows.net
DB_DATABASE=your-database-name
DB_USERNAME=your-username
DB_PASSWORD=your-password
DB_DRIVER=ODBC Driver 17 for SQL Server
```

## Usage

### Multi-Query Analysis

Run the main pipeline to analyze all queries:

```bash
python main.py
```

This will:
1. Load PMC IDs from `data/query_*_pmc_ids.txt` files
2. Fetch article content from the database
3. Remove stop words from queries and documents
4. Rank documents by semantic similarity
5. Generate SHAP explanations for top-ranked documents
6. Save results to `shap_outputs/query_*/`

### Manual Analysis

Analyze a single document:

```bash
python manual_shapley.py
```

This will analyze the text in `manual_shapley_data.txt` and save results to `manual_shap_outputs/`.

## Queries

The system analyzes three research queries:

1. **Query 1**: Impact of Advanced Cardiac Life Support courses on knowledge retention and competency in undergraduate medical education in the UK
2. **Query 2**: Impact of first-pass defibrillation on survival and neurological outcome in out-of-hospital ventricular fibrillation
3. **Query 3**: Effect of glucose administration or hyperglycemia management on outcomes after cardiac arrest

## Output Files

For each query, the system generates:

- `top_token_contributions.csv`: Token SHAP values ranked by contribution
- `{PMC_ID}_bar.png`: Bar chart visualization of top contributing tokens
- `{PMC_ID}_text.json`: Complete token-level explanation data

## Configuration

### Model Settings

- **Semantic Model**: `all-mpnet-base-v2` (SentenceTransformer)
- **SHAP Evaluations**: 500 (adjustable via `max_evals` parameter)
- **Top Documents**: 3 per query (adjustable via `top_k` parameter)

### Customization

- Modify queries in `QUERIES` dictionary in `main.py`
- Adjust `top_k` parameter to analyze more/fewer documents
- Change `limit_per_id` to fetch multiple articles per PMC ID
- Update `max_evals` for more/less detailed SHAP analysis

## SHAP Value Interpretation

- **Positive values**: Tokens that increase similarity to the query
- **Negative values**: Tokens that decrease similarity to the query
- **Magnitude**: Indicates the strength of contribution
- **Range**: Typically between -0.1 and +0.1 for cosine similarity

## Dependencies

- `sentence-transformers`: Semantic embeddings
- `shap`: Explainability analysis
- `numpy`: Numerical operations
- `pandas`: Data manipulation
- `sqlalchemy`: Database connectivity
- `matplotlib`: Visualization
- `tqdm`: Progress bars
- `python-dotenv`: Environment variables

## Hardware Requirements

- **CPU**: Multi-core recommended for faster encoding
- **Memory**: 8GB+ RAM recommended
- **GPU**: Optional CUDA-capable GPU for faster inference
- **Storage**: Varies based on number of documents

## Performance Notes

- SHAP analysis is computationally intensive
- Processing time scales with document length and `max_evals`
- GPU acceleration available for SentenceTransformer encoding
- Consider reducing `max_evals` for faster results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{dhs-dissertation-shapley,
  title={SHAP Explainability for Semantic Retrieval},
  author={Your Name},
  year={2025},
  url={https://github.com/mrk007Git/dhs-dissertation-shapley}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].