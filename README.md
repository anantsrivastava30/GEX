# Quant Analysis Project

This project analyzes market and options data and uses OpenAI APIs to generate trade strategies. It is built with a modular design, uses Streamlit for the UI, and Supabase as the backend database.

## Project Structure
- `quant.py` - Main application code with API calls and markdown report generation.
- `db.py` - Database interaction using Supabase.
- `helpers.py` - (Additional helper functions, if any.)

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. (Optional) Install as a package:
   ```
   python setup.py install
   ```
3. Set up your environment variables in a `.env` file or using Streamlit secrets.

## Running the Application

Run the project locally with:
```
streamlit run quant.py
```

## Testing

Tests can be added under a `tests/` directory; then run:
```
pytest
```

## CI

The repository includes a GitHub Actions workflow (`.github/workflows/ci.yml`) to run tests on every push.
