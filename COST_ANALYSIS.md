# Trimora — Cost & Expense Analysis

## Pipeline Cost Overview

Trimora uses a **rules-first, LLM-last** approach. 95% of processing is done with deterministic algorithms. LLM is only used for final verification of top candidates.

---

## API Pricing (as of 2025)

| Provider | Model | Input Cost (per 1M tokens) | Output Cost (per 1M tokens) | Speed |
|----------|-------|---------------------------|----------------------------|-------|
| **Groq** | Llama 3.3 70B | $0.59 | $0.79 | ~80 tok/s |
| **Groq** | Whisper large-v3 | $0.111 / minute | — | Real-time |
| OpenAI | GPT-4o | $2.50 | $10.00 | ~40 tok/s |
| OpenAI | Whisper | $0.006 / minute | — | Real-time |
| Anthropic | Claude 3.5 Sonnet | $3.00 | $15.00 | ~50 tok/s |
| Google | Gemini 1.5 Flash | $0.075 | $0.30 | ~100 tok/s |

**Trimora uses Groq for everything** — it's the cheapest and fastest option.

---

## Cost by Video Length

### Full LLM Approach (Sending Entire Transcript)

| Duration | Words | Tokens (Input) | Tokens (Output) | Groq Cost | GPT-4o Cost | Claude Cost |
|----------|-------|----------------|-----------------|-----------|-------------|-------------|
| 10 min | ~1,500 | ~2,000 | ~1,500 | $0.001 | $0.005 | $0.006 |
| 30 min | ~4,500 | ~6,000 | ~3,000 | $0.006 | $0.030 | $0.038 |
| 1 hour | ~9,000 | ~12,000 | ~3,500 | $0.010 | $0.065 | $0.083 |
| 2 hours | ~18,000 | ~24,000 | ~4,000 | $0.017 | $0.100 | $0.128 |
| 4 hours | ~36,000 | ~48,000 | ~4,500 | $0.032 | $0.165 | $0.208 |
| 5 hours | ~45,000 | ~60,000 | ~4,500 | $0.040 | $0.195 | $0.248 |

### Rules-First Approach (Trimora's Method)

| Duration | Words | Rules Cost | LLM Verification Cost | Total Cost |
|----------|-------|------------|----------------------|------------|
| 10 min | ~1,500 | $0.000 | $0.001 | **$0.001** |
| 30 min | ~4,500 | $0.000 | $0.002 | **$0.002** |
| 1 hour | ~9,000 | $0.000 | $0.003 | **$0.003** |
| 2 hours | ~18,000 | $0.000 | $0.004 | **$0.004** |
| 4 hours | ~36,000 | $0.000 | $0.005 | **$0.005** |
| 5 hours | ~45,000 | $0.000 | $0.005 | **$0.005** |

**Savings: 6-10x cheaper than full LLM approach.**

---

## Transcription Cost (Groq Whisper)

| Duration | Audio Minutes | Groq Cost | OpenAI Cost |
|----------|--------------|-----------|-------------|
| 10 min | 10 | $1.11 | $0.06 |
| 30 min | 30 | $3.33 | $0.18 |
| 1 hour | 60 | $6.66 | $0.36 |
| 2 hours | 120 | $13.32 | $0.72 |
| 4 hours | 240 | $26.64 | $1.44 |
| 5 hours | 300 | $33.30 | $1.80 |

**Note:** Groq Whisper is more expensive than OpenAI Whisper, but significantly faster (real-time vs 3-5x). For production, consider OpenAI Whisper for cost optimization.

---

## Total Cost Per Video (Trimora Approach)

| Duration | Transcription | Processing | LLM Verify | **Total** |
|----------|--------------|------------|------------|-----------|
| 10 min | $1.11 | $0.00 | $0.001 | **$1.11** |
| 30 min | $3.33 | $0.00 | $0.002 | **$3.33** |
| 1 hour | $6.66 | $0.00 | $0.003 | **$6.66** |
| 2 hours | $13.32 | $0.00 | $0.004 | **$13.32** |
| 4 hours | $26.64 | $0.00 | $0.005 | **$26.65** |
| 5 hours | $33.30 | $0.00 | $0.005 | **$33.31** |

---

## Monthly Cost Estimates

### Casual User (5 videos/month, 30 min each)

| Item | Cost |
|------|------|
| Transcription (5 × $3.33) | $16.65 |
| Processing | $0.00 |
| LLM verification (5 × $0.002) | $0.01 |
| **Monthly Total** | **$16.66** |

### Active Creator (20 videos/month, 1 hour each)

| Item | Cost |
|------|------|
| Transcription (20 × $6.66) | $133.20 |
| Processing | $0.00 |
| LLM verification (20 × $0.003) | $0.06 |
| **Monthly Total** | **$133.26** |

### Agency (100 videos/month, mixed lengths)

| Item | Cost |
|------|------|
| Transcription (100 avg 30 min) | $333.00 |
| Processing | $0.00 |
| LLM verification (100 × $0.002) | $0.20 |
| **Monthly Total** | **$333.20** |

---

## Free Tier Limits

| Provider | Free Tier | Monthly Limit |
|----------|-----------|---------------|
| Groq | Llama 3.3 70B | 14,400 min/month (~$8.50 value) |
| Groq | Whisper | 2 hours/day |
| Google Gemini | 1.5 Flash | 1,500 requests/day |

**For casual users:** Groq free tier covers ~4 hours of audio transcription per day.

---

## Cost Comparison with Competitors

| Tool | Monthly Cost | Videos | Cost per Video |
|------|-------------|--------|----------------|
| **Trimora (self-hosted)** | $17-133 | 5-20 | **$3.33** |
| Opus Clip | $15-228 | Unlimited | $0.75-11.40 |
| Kapwing | $24-79 | 10-100 | $0.79-7.90 |
| Descript | $24-33 | 20-50 | $0.48-1.65 |
| HeyGen | $24-180 | 10-60 | $0.40-18.00 |

**Trimora advantage:** No subscription, no watermarks, full data privacy, unlimited clips.

---

## Hidden Costs (Not in Trimora)

| Item | Cost | Notes |
|------|------|-------|
| Server (Vercel) | $0-20/mo | Free tier sufficient for casual use |
| Server (self-hosted) | $0 | Runs on your machine |
| Storage | $0-5/mo | Output clips, temporary files |
| Domain (optional) | $10-15/year | trimora.app |

---

## ROI Analysis

### For Content Creators

| Metric | Value |
|--------|-------|
| Time saved per video | 2-4 hours (manual clipping) |
| Cost per clip (Trimora) | $0.005 (processing only) |
| Cost per clip (manual) | $50-200 (VA or freelancer) |
| Break-even | 1 video/month |

### For Agencies

| Metric | Value |
|--------|-------|
| Time saved per video | 3-5 hours |
| Cost per clip (Trimora) | $3.33 |
| Cost per clip (manual) | $100-500 |
| Monthly savings (20 videos) | $1,933-$9,933 |

---

## Summary

- **Transcription is the main cost** (99% of total)
- **Processing is free** (deterministic algorithms)
- **LLM verification is negligible** ($0.002-0.005 per video)
- **Self-hosted = no subscription fees**
- **Scales linearly with video length**
- **30-minute videos cost ~$3.33 total**
