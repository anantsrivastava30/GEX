---
title: GEX Options Analytics API
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# GEX Options Analytics API

FastAPI backend for the GEX dealer-positioning platform (net GEX, gamma-gap,
skew, unusual-activity proxy, and own-data market history).

This file is the Hugging Face Space configuration card. When the Space is
built from this repository it is copied to the Space root as `README.md`;
the YAML frontmatter above tells Hugging Face to build the root `Dockerfile`
and route traffic to port 7860.

Health check once live: `GET /api/health`
OpenAPI docs: `GET /docs`

Optional environment variables (set as Space **Secrets**):

- `TRADIER_TOKEN` — enables live option-chain data (omit for graceful, empty responses).
- `OPENAI_API_KEY` — enables AI narratives when that router lands.
- `CORS_ORIGINS` — JSON array of allowed origins, e.g. `["https://your-frontend.vercel.app"]`.
