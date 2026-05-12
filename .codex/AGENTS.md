# Codex Instructions

This directory is the Exodus Chat Escape game.

## Scope

Work primarily in this directory. Do not stage or commit `weekly_market_report/` files while working on the Exodus branch.

## Project Shape

- `index.html`: static game UI, data, styles, and browser logic
- `assets/`: background and animated NPC frame images
- `local-server.js`: local static server
- `netlify.toml`: static deployment config
- `README.md`: run instructions and asset contract

## Checks

Run syntax checks after JavaScript/server changes:

```bash
npm run check
```

Run the local server for UI or gameplay changes:

```bash
npm run dev
```

Expected local URL:

```text
http://127.0.0.1:4174/
```

## Commit Hygiene

Before committing:

```bash
git diff --cached --name-status
```

The staged list should only include Exodus game files and Exodus-specific instruction files.
