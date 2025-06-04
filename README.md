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

### Securing AI Access

Access to the AI analysis tab now relies on user accounts rather than a single
token. Define approved usernames and password hashes in `AI_USERS` within your
Streamlit secrets or environment. Example `secrets.toml` snippet:

```toml
[AI_USERS]
alice = "<sha256-of-password>"
bob   = "<sha256-of-password>"
```

When a user logs in with their username and password, the SHA‑256 hash of the
password is compared against this table. Only authenticated users can run the
model, giving you fine‑grained control over who may access it. If no users are
configured, the AI tab is hidden and the dashboard still functions normally.
The logged-in user is shown in the sidebar so you know who is active. When a
non‑authenticated user clicks **Run AI Analysis**, a login dialog pops up asking
for their credentials.

## Running the Application

Run the project locally with:
```
streamlit run quant.py
```

The sidebar displays the short commit hash of the current version so you can
track exactly which build is running.

### Optional Thread Debugging

Set the environment variable `THREAD_DEBUG=1` before running the app to log
thread start and completion. Runtime, CPU, and memory deltas are printed for
each thread so you can gauge resource usage. Debugging is enabled automatically
via `helpers.py` so any module that imports it will patch the threading layer.

## Testing

Tests can be added under a `tests/` directory; then run:
```
pytest
```

## CI

The repository includes a GitHub Actions workflow (`.github/workflows/ci.yml`) to run tests on every push.
