# Kindred Paws — dog walking & 24/7 pet care (NLP class project)

This is a landing page for a dog walking business. It has two natural language processing features, and both run entirely in the browser:

1. **"Tell us about your dog" analyzer.** You describe your dog and schedule in plain English. The page pulls out details using patterns (name, breed, age, schedule, health needs, travel). It reads temperament by comparing your words to trait words with embedding similarity. Then it recommends a plan by comparing the meaning of your description to each plan's description (cosine similarity of averaged word vectors).
2. **Paws & Words lab.** Type a word to see its 10 nearest neighbors in 50-dimension GloVe word-embedding space (a word2vec-family model). The words are plotted in 2D and 3D with PCA, and there's an analogy solver (a : b :: c : ?).

## Files
- `index.html`, `styles.css`, `app.js` — the site
- `data/vocab.json`, `data/vectors.bin` — 20,087 words × 50 dimensions from GloVe (Wikipedia + Gigaword 6B), stored as 8-bit numbers to keep the files small (~1.2 MB total)
- `docs/business-kit.md` — customer avatar, diary entries, brand guide, and landing page requirements
- `netlify.toml` — Netlify config (static site, no build step)

## Deploy
Netlify → Add new project → Import from GitHub → pick this repo → Deploy. No build command is needed; the publish directory is `.`.

Word vectors: GloVe, Pennington, Socher & Manning (2014), Stanford NLP. Public Domain Dedication and License (PDDL).
