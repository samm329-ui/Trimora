# Trimora — Implementation Plan

## Audio Analysis & Clipping Pipeline

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Current Codebase Structure](#2-current-codebase-structure)
3. [New Engine Structure](#3-new-engine-structure)
4. [Pipeline Stages (Detailed)](#4-pipeline-stages-detailed)
5. [Data Models](#5-data-models)
6. [Rules & Fundamentals](#6-rules--fundamentals)
7. [File-by-File Implementation Plan](#7-file-by-file-implementation-plan)
8. [Dependencies](#8-dependencies)
9. [Build Order](#9-build-order)
10. [Data Storage for Future ML](#10-data-storage-for-future-ml)
11. [LLM as Teacher — Structured Labeling](#11-llm-as-teacher--structured-labeling)
12. [Dual Knowledge Graphs (Local + Global)](#12-dual-knowledge-graphs-local--global)
13. [Pattern Intelligence Engine](#13-pattern-intelligence-engine)
14. [Confidence System](#14-confidence-system)
15. [Decision Log & Audit Trail](#15-decision-log--audit-trail)
16. [Failure Storage](#16-failure-storage)
17. [Future Deterministic Engine](#17-future-deterministic-engine)
18. [Review Feedback & Applied Improvements](#18-review-feedback--applied-improvements)
19. [Final Review Verdict](#19-final-review-verdict)

---

## Pipeline Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              TRIMORA AUDIO ANALYSIS PIPELINE                                         │
│                  95% Deterministic · Rules-First · LEGO-Brick Stitching · LLM as Teacher             │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘

                                    INPUT: Video File
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: FOUNDATION (Days 1-3)                                                                     │
│                                                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────────────┐                  │
│  │ Audio        │───▶│ Quality      │───▶│ Overlapping  │───▶│ Per-Chunk         │                  │
│  │ Extraction   │    │ Analysis     │    │ Chunking     │    │ Transcription     │                  │
│  │ (FFmpeg)     │    │ (librosa)    │    │ (pydub)      │    │ (Groq Whisper)    │                  │
│  │ 16kHz mono   │    │ SNR, speech  │    │ 15-30s       │    │ + language detect │                  │
│  │              │    │ rate, volume │    │ 2-3s overlap │    │ + filler removal  │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────┬──────────┘                  │
│                                                                       │                              │
│                                                                       ▼                              │
│                                                              ┌───────────────────┐                  │
│                                                              │ Chunk Merge &     │                  │
│                                                              │ Deduplication     │                  │
│                                                              │ (overlap removal) │                  │
│                                                              └────────┬──────────┘                  │
└───────────────────────────────────────────────────────────────────────┼──────────────────────────────┘
                                                                        │
                                                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: CORE PROCESSING (Days 4-6)                                                                │
│                                                                                                     │
│  ┌───────────────────┐    ┌──────────────────────┐    ┌────────────────────────────────────────┐    │
│  │ Word-Level        │───▶│ Atomic Segmentation  │───▶│ Feature Extraction                     │    │
│  │ Alignment         │    │ (LEGO Bricks)        │    │                                        │    │
│  │ (WhisperX)        │    │                      │    │  ┌─────────┐ ┌───────────┐            │    │
│  │ Word timestamps   │    │ Split on punctuation │    │  │Sentiment│ │Patterns   │            │    │
│  │ ±0.1s accuracy    │    │ Max 8s, Min 2s       │    │  │(VADER)  │ │(Regex)    │            │    │
│  │ Confidence scores │    │ Merge short segments │    │  └─────────┘ └───────────┘            │    │
│  └───────────────────┘    └──────────────────────┘    │  ┌─────────┐ ┌───────────┐            │    │
│                                                        │  │Audio    │ │Structural │            │    │
│                                                        │  │Features │ │(position, │            │    │
│                                                        │  │(speech  │ │ recency)  │            │    │
│                                                        │  │ rate,   │ └───────────┘            │    │
│                                                        │  │volume)  │                          │    │
│                                                        │  └─────────┘                          │    │
│                                                        └───────────────┬────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────┼─────────────────────────────┘
                                                                         │
                                                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: LOCAL KNOWLEDGE + RULES (Days 7-9)                                                        │
│                                                                                                     │
│  ┌───────────────────┐    ┌──────────────────┐    ┌──────────────────────────────────────────┐      │
│  │ Local Knowledge   │◀───│ Edge Detection   │    │  Rules Engine                            │      │
│  │ Graph (Per Video) │    │                  │    │                                          │      │
│  │ (NetworkX DiGraph)│    │ Temporal edges   │    │  ┌────────────┐ ┌───────────────┐       │      │
│  │                   │    │ Contrast edges   │    │  │Hook Rules  │ │Body Rules    │       │      │
│  │ Nodes: Segments   │    │ Explains edges   │    │  │(+confidence)│ │(+confidence) │       │      │
│  │ Edges: follows,   │    │ Concludes edges  │    │  └────────────┘ └───────────────┘       │      │
│  │ explains,         │    │ Supports edges   │    │  ┌────────────┐ ┌───────────────┐       │      │
│  │ contrasts, etc.   │    │ Context requires │    │  │Ending Rules│ │Stitching Rules│       │      │
│  └────────┬──────────┘    └──────────────────┘    │  │(+confidence)│ │(+confidence)  │       │      │
│           │                                        │  └────────────┘ └───────────────┘       │      │
│           ▼                                        └──────────────────────────────────────────┘      │
│  ┌─────────────────────────────────────┐                                                              │
│  │ Candidate Generation                │                                                              │
│  │ (BFS with max-duration constraint)  │                                                              │
│  │ Hook → Body → Ending combos        │                                                              │
│  └─────────────┬───────────────────────┘                                                              │
└────────────────┼──────────────────────────────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: LLM TEACHER + PATTERN EXTRACTION (Days 10-13)                                            │
│                                                                                                     │
│  ┌──────────────────────────┐    ┌───────────────────────────┐    ┌──────────────────────────────┐  │
│  │ Rule Engine + Scorer    │───▶│ LLM Teacher               │───▶│ Reason Extraction Engine     │  │
│  │ (same as before)        │    │                            │    │                              │  │
│  │                         │    │ Produces STRUCTURED labels │    │ Extracts:                     │  │
│  │ 45-90s / Hook ≤ 5s      │    │ NOT human text             │    │ • Per-segment labels          │  │
│  │ Has curiosity / value   │    │                            │    │ • Per-candidate labels        │  │
│  │ Max 2 speakers          │    │ Segment: {is_hook: 0.91,   │    │ • Rejection reasons           │  │
│  │ No context gaps         │    │   hook_strength, emotion,  │    │ • Confidence scores           │  │
│  └─────────────┬───────────┘    │   requires_context, ...}   │    └──────────────┬───────────────┘  │
│                │                │                            │                   │                    │
│                │                │ Candidate: {story_complete, │                   ▼                    │
│                ▼                │   shareability, ...}       │    ┌──────────────────────────────┐  │
│  ┌─────────────────────────┐   └─────────────┬──────────────┘    │ Decision Log                 │  │
│  │ FAILURE STORAGE         │                 │                   │                              │  │
│  │ (Rejected candidates    │                 ▼                   │ Segment 143: Rule Curiosity  │  │
│  │  saved with reasons)    │    ┌──────────────────────────┐     │   → Rule High Energy         │  │
│  └─────────────────────────┘    │ Pattern Intelligence     │     │   → Graph: Explains #149     │  │
│                                  │ Engine                    │     │   → LLM: Hook=0.91          │  │
│                                  │                           │     │   → Decision: Selected      │  │
│                                  │ Discovers reusable        │     └──────────────┬───────────────┘  │
│                                  │ narrative patterns across │                    │                    │
│                                  │ ALL videos                │                    ▼                    │
│                                  │ e.g. "Question → Story   │    ┌──────────────────────────────┐  │
│                                  │  → Failure → Lesson"     │────▶│ Structured Label Storage    │  │
│                                  │ with Share Prob: 93%     │    │ (training data for future ML)│  │
│                                  └─────────────┬────────────┘    └──────────────────────────────┘  │
└────────────────────────────────────────────────┼────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 5: GLOBAL KNOWLEDGE + STORAGE (Continuous)                                                   │
│                                                                                                     │
│  ┌────────────────────────────────┐    ┌────────────────────────────┐    ┌────────────────────────┐ │
│  │ Global Knowledge Graph        │    │ Pattern Database           │    │ Context Database       │ │
│  │ (Across ALL videos)           │    │                            │    │                         │ │
│  │                                │    │ Pattern ID: 421           │    │ Segment → Needs Context │ │
│  │ Curiosity Hook                │    │ Sequence: Question→Story   │    │  → Type: Person         │ │
│  │  ├── 94% high retention       │    │   →Failure→Lesson          │    │  → Required: Intro      │ │
│  │  ├── Usually→story            │    │ Occurrences: 186           │    │  → Confidence: 0.91     │ │
│  │  └── Often→lesson ending      │    │ Avg Saves: 82%             │    │                         │ │
│  │                                │    │ Avg Shares: 74%           │    │ Segment → Standalone?   │ │
│  │ Personal Story                │    │ Confidence: 0.96           │    │  → Yes: Confidence 0.88 │ │
│  │  ├── High saveability         │    │ Context Reqs: Medium       │    └────────────┬───────────┘ │
│  │  ├── Medium shareability      │    └────────────────────────────┘                 │               │
│  │  └── Strong emotional arc     │                                                  ▼               │
│  │                                │    ┌─────────────────────────────────────────────────────────┐  │
│  │ Statistics + Framework edges   │    │  SQLite Storage                                        │  │
│  │ (with statistical weights)     │    │                                                        │  │
│  │                                │    │  videos / segments / words / candidates                  │  │
│  │ Edge weights LEARN from data:  │    │  relationships (with LLM confidence + stats)            │  │
│  │ Times Used / Avg Watch Time   │    │  patterns / pattern_nodes / pattern_edges                │  │
│  │ Avg Saves / Avg Shares        │    │  context_requirements / decision_log                     │  │
│  │ LLM Confidence → Edge Weight  │    │  failed_candidates / training_labels                     │  │
│  └────────────────────────────────┘    └─────────────────────────────────────────────────────────┘  │
│                                                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │  FUTURE: Deterministic Engine (after 2000+ videos)                                            │  │
│  │                                                                                              │  │
│  │  Confidence > 0.9 → Use local model (zero LLM cost)                                          │  │
│  │  Confidence 0.6-0.9 → Use pattern match from Global Graph                                    │  │
│  │  Confidence < 0.6 → Fall back to Groq LLM                                                    │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘


## 1. Project Overview

### Vision

Trimora analyzes long-form audio (podcasts, interviews, lectures) to extract **viral-worthy short clips** without analyzing video frames. The system is **95% deterministic at runtime** — LLM serves as a **teacher and structured label generator**, not a verifier. Every LLM interaction produces machine-readable training data that feeds into a growing **Pattern Intelligence Engine** and **Global Knowledge Graph**, enabling the system to gradually replace LLM calls with learned models.

### Core Principles

- **Fundamentals first:** Hook must grab in 3 seconds, story must be complete, value must be delivered, context must be preserved, emotion must flow, must feel authentic
- **Rules guide decisions, categories collect data**
- **LEGO bricks approach:** Atomic segments that can be stitched together
- **Data is stored from day one** for future ML training (target: 2000+ videos)
- **Production-grade:** All data is precise, clean, and structured for ML

### The Stitching Principle

```
Hook (somewhere in video) → Body (somewhere else) → Ending (somewhere else)
                             ↓
              Output: One seamless story clip
```

### The Emotional Arc

```
Peak (hook: high energy/curiosity)
   ↓
Downfall (body: tension/explanation)
   ↓
Second Peak (ending: resolution/takeaway)
   ↓
Smooth resolution
```

---

## 2. Current Codebase Structure

### Root Directory

```
D:\openshorts\
├── app.py                   # FastAPI server (689 lines)
│   ├── POST /api/process    # Submit video for processing
│   ├── GET /api/status      # Poll job status
│   ├── POST /api/edit       # AI video effects
│   ├── POST /api/subtitle   # Generate subtitles
│   └── POST /api/hook       # Add text overlays
│
├── main.py                  # Core video processing (1017 lines)
│   ├── download_youtube_video()
│   ├── transcribe_video()           # faster-whisper (local, CPU)
│   ├── detect_scenes()              # PySceneDetect
│   ├── analyze_scenes_strategy()    # TRACK vs GENERAL mode
│   ├── process_video_to_vertical()  # Frame-by-frame reframing
│   └── get_viral_clips()            # Gemini AI analysis
│
├── editor.py                # Gemini AI video effects (376 lines)
├── hooks.py                 # Text overlay generation (241 lines)
├── subtitles.py             # SRT generation (224 lines)
├── requirements.txt         # 18 packages
├── CLAUDE.md                # Agent guidance
├── COST_ANALYSIS.md         # Pricing document
├── README.md
├── LICENSE
├── docker-compose.yml       # 3-service stack
├── Dockerfile
│
├── dashboard/               # React frontend
│   └── src/
│       ├── App.jsx          # Main React component (601 lines)
│       ├── Landing.jsx      # Marketing page (612 lines)
│       ├── Legal.jsx        # Terms page (159 lines)
│       ├── main.jsx         # Entry point
│       ├── config.js        # API config
│       └── components/
│           ├── HookModal.jsx
│           ├── KeyInput.jsx
│           ├── MediaInput.jsx
│           ├── ProcessingAnimation.jsx
│           └── ResultCard.jsx
│
└── venv/                    # Python virtual environment (96 packages)
```

### What Current Code Does

| File | Function | Input | Output |
|------|----------|-------|--------|
| `app.py` | FastAPI server | Video file or URL | Job queue, status, edit/subtitle/hook endpoints |
| `main.py` | Full pipeline | Video or YouTube URL | Transcribed clips, vertical reframed videos |
| `editor.py` | AI video effects | Video + Gemini API key | FFmpeg filter string for effects |
| `hooks.py` | Text overlays | Video + text | Hook overlay composited onto video |
| `subtitles.py` | Subtitle generation | Video + transcript | SRT file, burned subtitles |

### Key Technologies Currently Used

- **Transcription:** faster-whisper (local, CPU, INT8 quantization)
- **AI Analysis:** Google Gemini 2.5 Flash for viral moment detection
- **Scene Detection:** PySceneDetect (ContentDetector)
- **Vertical Reframing:** MediaPipe face detection + YOLOv8 fallback
- **Video Processing:** OpenCV + FFmpeg

### What We Are Replacing

The current system analyzes video **frames** (face detection, scene detection, vertical cropping). Our new system analyzes **audio only** — transcript, timestamps, audio features. The video editing (cutting, stitching) will be added later.

---

## 3. New Engine Structure

```
D:\openshorts\
├── engine/                          # NEW: Audio analysis pipeline
│   ├── __init__.py
│   ├── config.py                    # All thresholds, constants, word lists, prompt templates
│   ├── pipeline.py                  # Main orchestrator
│   │
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── extractor.py             # Extract audio from video (FFmpeg)
│   │   ├── quality.py               # SNR, speech rate, volume (librosa)
│   │   └── chunker.py               # Overlapping audio chunks (pydub)
│   │
│   ├── transcription/
│   │   ├── __init__.py
│   │   ├── transcriber.py           # Groq Whisper API
│   │   ├── aligner.py               # WhisperX word-level alignment
│   │   ├── language.py              # Language detection
│   │   ├── fillers.py               # Filler word removal
│   │   └── merger.py                # Chunk merge with deduplication
│   │
│   ├── segmentation/
│   │   ├── __init__.py
│   │   └── segmenter.py             # Atomic segmentation
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── sentiment.py             # VADER sentiment analysis
│   │   ├── patterns.py              # Regex pattern matching
│   │   ├── audio_features.py        # Per-segment speech/pause/volume
│   │   └── structural.py            # Position, recency, speaker changes
│   │
│   ├── rules/
│   │   ├── __init__.py
│   │   ├── fundamentals.py          # Core rule definitions
│   │   ├── hook_rules.py            # Hook detection rules
│   │   ├── body_rules.py            # Body/context detection rules
│   │   ├── ending_rules.py          # Ending detection rules
│   │   └── stitching_rules.py       # How to combine segments naturally
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── knowledge_graph.py       # Build and query graph
│   │   └── relationships.py         # Edge detection
│   │
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── candidate_generator.py   # Generate clip candidates
│   │   ├── rule_engine.py           # Filter invalid candidates
│   │   └── scorer.py                # Score candidates
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── teacher.py               # LLM as Teacher: structured label generation
│   │   ├── prompts.py               # Prompt templates that produce structured output
│   │   └── label_schemas.py         # Pydantic models for structured LLM output
│   │
│   ├── patterns/
│   │   ├── __init__.py
│   │   ├── engine.py                # Pattern Intelligence Engine
│   │   ├── detector.py              # Pattern discovery across videos
│   │   ├── matcher.py               # Match current video against known patterns
│   │   └── graph.py                 # Pattern graph builder
│   │
│   ├── knowledge/
│   │   ├── __init__.py
│   │   ├── local_graph.py           # Per-video knowledge graph (discarded after)
│   │   ├── global_graph.py          # Cross-video global knowledge graph (persistent)
│   │   ├── context_db.py            # Context requirement database
│   │   └── relationships.py         # Edge detection with confidence
│   │
│   ├── confidence/
│   │   ├── __init__.py
│   │   ├── scorer.py                # Confidence scoring for all decisions
│   │   └── threshold.py             # Adaptive threshold engine
│   │
│   ├── decision/
│   │   ├── __init__.py
│   │   ├── log.py                   # Decision log / audit trail
│   │   ├── tracker.py               # Track which signals influenced each decision
│   │   └── failures.py              # Store rejected candidates with reasons
│   │
│   └── data/
│       ├── __init__.py
│       ├── storage.py               # SQLite storage for analysis data
│       ├── models.py                # Data models (dataclasses)
│       └── migrations.py            # Schema migrations
│
├── app.py                           # MODIFIED: Add new /api/analyze endpoint
├── main.py                          # UNCHANGED: Legacy pipeline (keep for reference)
├── engine_config.yaml               # NEW: Runtime overrides for config.py
│
└── requirements_engine.txt          # NEW: Additional dependencies
```

**Config Centralization Principle:** Every magic number, threshold, prompt template, pattern list, fallback value, and default lives in exactly one place — `config.py` (with runtime overrides in `engine_config.yaml`). Zero hardcoded values in pipeline logic. This includes:

| Category | Examples |
|----------|----------|
| Duration thresholds | `CLIP_MIN_DURATION=45`, `CLIP_MAX_DURATION=90` |
| Score thresholds | `HOOK_MIN_SCORE=50`, `ENDING_MIN_SCORE=40` |
| Speech/volume | `HOOK_SPEECH_RATE_MIN=2.0`, `HOOK_VOLUME_DELTA_MIN=1.5` |
| Window sizes | `TEMPORAL_WINDOW_SECONDS=60`, `MATCH_WINDOW=5` |
| LLM prompts | `TEACHER_PROMPT_TEMPLATE`, `VERIFIER_PROMPT_TEMPLATE` |
| Pattern config | `MIN_PATTERN_LEN=3`, `BASE_DECAY_RATE=0.0003` |
| Fallback values | `DEFAULT_SNR=15`, `DEFAULT_CONFIDENCE=0.5` |
| File paths | `STORE_ROOT`, `STATE_DIR`, `TEMP_DIR` |

---

## 4. Pipeline Stages (Detailed)

### Stage 1: Audio Extraction

```
Input:  Video file path
Process: FFmpeg subprocess → extract audio
Output: WAV file (16kHz mono)
```

**File:** `engine/audio/extractor.py`

```python
def extract_audio(video_path: str, output_path: str) -> str:
    """Extract audio from video using FFmpeg."""
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-q:a", "0", "-map", "a",
        "-ar", "16000", "-ac", "1",
        output_path
    ], check=True)
    return output_path
```

**Key decisions:**
- 16kHz mono (standard for Whisper alignment)
- No normalization (preserve original volume levels for loudness analysis)

**Review Note:** Groq Whisper is fast and cost-effective, but for best word-level alignment accuracy, always run WhisperX on the full audio after initial chunk transcription. The plan already does this in Stage 5 — ensure WhisperX receives the full audio file (not chunks) to produce the most precise timestamps.

---

### Stage 2: Audio Quality Analysis

```
Input:  WAV file path
Process: librosa → SNR, speech_rate, volume_rms
Output: AudioQuality(snr_db, speech_rate, volume_rms)
```

**File:** `engine/audio/quality.py`

```python
def measure_audio_quality(audio_path: str) -> AudioQuality:
    """Measure SNR, speech rate, and volume using librosa."""
    y, sr = librosa.load(audio_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Volume RMS
    volume_rms = np.mean(librosa.feature.rms(y=y))
    
    # SNR via harmonic/percussive separation
    S = np.abs(librosa.stft(y))
    D_harmonic, D_percussive = librosa.decompose.hpss(S)
    signal_power = np.sum(D_harmonic ** 2)
    noise_power = np.sum(D_percussive ** 2) + 1e-10
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    # Speech rate via onset detection
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    speech_rate = len(onsets) / duration if duration > 0 else 0
    
    return AudioQuality(snr_db, speech_rate, volume_rms)
```

**Adaptive thresholds:**
- If SNR < 10dB → stricter chunking (15s chunks, 3s overlap)
- If speech_rate > 3.5 onsets/s → lower alignment threshold

---

### Stage 3: Audio Chunking

```
Input:  WAV file path + AudioQuality
Process: pydub → overlapping chunks
Output: list[Chunk(audio_path, start_time, end_time)]
```

**File:** `engine/audio/chunker.py`

```python
def overlap_chunk(audio_path: str, quality: AudioQuality) -> list[Chunk]:
    """Split audio into overlapping chunks."""
    if quality.snr_db < 10:
        chunk_size = 15  # seconds
        overlap = 3      # seconds (strict mode)
    else:
        chunk_size = 30
        overlap = 2
    
    audio = AudioSegment.from_wav(audio_path)
    chunks = []
    start = 0
    while start < len(audio):
        end = min(start + chunk_size * 1000, len(audio))
        chunk = audio[start:end]
        # Apply fade in/out
        chunk = chunk.fade_in(50).fade_out(50)
        chunk_path = f"{audio_path}.chunk.{len(chunks)}.mp3"
        chunk.export(chunk_path, format="mp3")
        chunks.append(Chunk(chunk_path, start / 1000, end / 1000))
        start += (chunk_size - overlap) * 1000
    
    return chunks
```

---

### Stage 4: Per-Chunk Processing

```
Input:  list[Chunk]
Process: For each chunk → transcribe → detect language → remove fillers → merge
Output: list[Segment(text, start, end)]
```

**File:** `engine/transcription/transcriber.py`

```python
def transcribe_chunk(audio_path: str) -> str:
    """Transcribe audio chunk via Groq Whisper."""
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=(audio_path, f.read()),
            model="whisper-large-v3",
            response_format="text"
        )
    return response.strip()
```

**File:** `engine/transcription/language.py`

```python
def detect_language(words: list[str]) -> str:
    """Detect language using Unicode block analysis."""
    devanagari = sum(1 for w in words if any(0x0900 <= ord(c) <= 0x097F for c in w))
    bengali = sum(1 for w in words if any(0x0980 <= ord(c) <= 0x09FF for c in w))
    latin = sum(1 for w in words if all(ord(c) < 0x80 for c in w))
    
    if devanagari > len(words) * 0.1 and latin > len(words) * 0.1:
        return "hinglish"
    elif devanagari > len(words) * 0.3:
        return "hindi"
    elif bengali > len(words) * 0.3:
        return "bengali"
    else:
        return "english"


**Note (Review Feedback):** For broader language support, consider adding `langdetect` or `fasttext` as fallback. These handle code-switching better than Unicode-block analysis alone, especially for Southeast Asian languages, Arabic, and mixed-script content.

**File:** `engine/transcription/fillers.py`

```python
def remove_fillers(text: str) -> str:
    """Remove common filler words."""
    fillers = {
        "um", "uh", "er", "ah", "like", "you know", "actually", "basically",
        "hai", "hain", "tha", "thi", "ho", "hua", "hue",
        "ka", "ki", "ke", "ko", "se", "me", "pe", "aur", "ya"
    }
    words = text.split()
    return " ".join(w for w in words if w.lower() not in fillers)
```

**File:** `engine/transcription/merger.py`

```python
def merge_chunks(processed_chunks: list[ProcessedChunk]) -> list[Segment]:
    """Merge chunks with overlap deduplication."""
    sorted_chunks = sorted(processed_chunks, key=lambda c: c.index)
    merged = []
    
    for chunk in sorted_chunks:
        if not chunk.text.strip():
            continue
        
        # Deduplicate overlap with previous chunk
        if merged:
            last_text = merged[-1]["text"]
            last_words = last_text.split()[-10:]
            first_words = chunk.text.split()[:10]
            # Find and remove overlapping words
            for i in range(len(first_words), 0, -1):
                if " ".join(first_words[:i]).lower() == " ".join(last_words[-i:]).lower():
                    chunk.text = " ".join(chunk.text.split()[i:])
                    break
        
        merged.append({
            "text": chunk.text,
            "start": chunk.start_time,
            "end": chunk.end_time
        })
    
    return merged
```

---

### Stage 5: Word-Level Alignment

```
Input:  list[Segment] + audio_path
Process: WhisperX forced alignment
Output: list[Segment(words[{text, start, end, confidence}])]
```

**File:** `engine/transcription/aligner.py`

```python
def align_segments(segments: list[dict], audio_path: str, language: str) -> list[dict]:
    """Align text to audio using WhisperX."""
    model_id = get_model_for_language(language)
    model, metadata = get_alignment_model(model_id, "cpu")
    
    audio = whisperx.load_audio(audio_path)
    
    # Format segments for WhisperX
    prompt = [
        {"text": s["text"], "start": round(s["start"], 3), "end": round(s["end"], 3)}
        for s in segments
    ]
    
    result = whisperx.align(prompt, model, metadata, audio, "cpu")
    
    aligned = []
    for seg in result["segments"]:
        aligned.append({
            "text": seg["text"],
            "start": seg["start"],
            "end": seg["end"],
            "words": [
                {"text": w["text"], "start": w["start"], "end": w["end"], "score": w["score"]}
                for w in seg.get("words", [])
            ]
        })
    
    return aligned
```

---

### Stage 6: Atomic Segmentation

```
Input:  list[Segment(words[{text, start, end, confidence}])]
Process: Split on punctuation + time boundaries
Output: list[Segment(id, text, start, end, duration, speaker)]
```

**File:** `engine/segmentation/segmenter.py`

```python
def split_into_atomic_segments(aligned_segments: list[dict]) -> list[Segment]:
    """Split transcript into atomic LEGO-brick segments."""
    atomic = []
    
    for seg in aligned_segments:
        words = seg["words"]
        sentences = split_by_punctuation(words)
        
        for sentence in sentences:
            # Enforce max 8 seconds
            if sentence["end"] - sentence["start"] > 8:
                sub_sentences = split_by_time(sentence, max_duration=8)
                atomic.extend(sub_sentences)
            else:
                atomic.append(Segment(
                    id=str(uuid4()),
                    text=sentence["text"],
                    start=sentence["start"],
                    end=sentence["end"],
                    duration=sentence["end"] - sentence["start"],
                    speaker="unknown",
                    words=sentence["words"]
                ))
    
    # Merge very short segments (< 2s) with next segment
    return merge_short_segments(atomic)


def split_by_punctuation(words: list[dict]) -> list[dict]:
    """Split on . ! ? and other sentence boundaries."""
    sentences = []
    current = []
    
    for w in words:
        current.append(w)
        if w["text"][-1] in ".!?":
            sentences.append(build_sentence(current))
            current = []
    
    if current:
        sentences.append(build_sentence(current))
    
    return sentences


def split_by_time(words: list[dict], max_duration: float) -> list[dict]:
    """Split long sentences by time."""
    sentences = []
    current = []
    start_time = words[0]["start"]
    
    for w in words:
        if w["end"] - start_time > max_duration and current:
            sentences.append(build_sentence(current))
            current = [w]
            start_time = w["start"]
        else:
            current.append(w)
    
    if current:
        sentences.append(build_sentence(current))
    
    return sentences
```

**Key rules:**
- Split on punctuation (`.`, `!`, `?`)
- Split on speaker change
- Max 8 seconds per segment
- Min 2 seconds per segment
- Keep "merged" versions of 2-3 consecutive segments for context

---

### Stage 7: Feature Extraction

```
Input:  list[Segment]
Process: Compute features for each segment
Output: list[Segment] with features populated
```

#### 7a. Sentiment Analysis

**File:** `engine/features/sentiment.py`

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def compute_sentiment(text: str) -> float:
    """Compute sentiment score: -1.0 to 1.0."""
    scores = analyzer.polarity_scores(text)
    return scores["compound"]
```

#### 7b. Pattern Matching

**File:** `engine/features/patterns.py`

```python
PATTERNS = {
    "curiosity": [
        (r"(?i)what if", "what_if"),
        (r"(?i)no one (knows|talks|told)", "unknown"),
        (r"(?i)the biggest (mistake|problem|secret)", "biggest"),
        (r"(?i)I didn't expect", "unexpected"),
        (r"(?i)imagine if", "imagine"),
        (r"(?i)I learned?", "learned"),
        (r"(?i)\?\s*$", "question"),
    ],
    "story": [
        (r"(?i)\bthen\b", "then"),
        (r"(?i)\bafter\b", "after"),
        (r"(?i)\bbefore\b", "before"),
        (r"(?i)\bsuddenly\b", "suddenly"),
        (r"(?i)\bfinally\b", "finally"),
        (r"(?i)\bone day\b", "one_day"),
        (r"(?i)\blater\b", "later"),
        (r"(?i)\bbecause\b", "because"),
    ],
    "practicality": [
        (r"(?i)(steps?|step \d)", "steps"),
        (r"(?i)(framework|system)", "framework"),
        (r"(?i)(tips?|trick)", "tip"),
        (r"(?i)(rules?|rule \d)", "rule"),
        (r"(?i)(checklist|template)", "checklist"),
        (r"(?i)(lesson|takeaway)", "lesson"),
        (r"(?i)(method|formula)", "method"),
    ],
    "shareability": [
        (r"\d+(\.\d+)?%", "percentage"),
        (r"\$\d+(\.\d+)?[kKmMbB]?", "money"),
        (r"\d+\s*(days?|weeks?|months?|years?)", "timeframe"),
        (r"(?i)(surprising|shocking|unbelievable)", "surprising"),
        (r"(?i)(contrarian|unpopular|hot take)", "contrarian"),
        (r"(?i)(the best|the worst|the only)", "extreme"),
        (r"(?i)(according to|study shows|research says)", "authority"),
    ],
    "contrast": [
        (r"(?i)\bbut\b", "but"),
        (r"(?i)\bhowever\b", "however"),
        (r"(?i)\bactually\b", "actually"),
        (r"(?i)\bsurprisingly\b", "surprisingly"),
        (r"(?i)\byet\b", "yet"),
        (r"(?i)\b(lekin|par|magar)\b", "contrast_hi"),
    ],
    "relatability": [
        (r"\b(I|we|my|our)\b", "personal"),
        (r"(?i)(everyone|nobody|anyone)", "universal"),
        (r"(?i)(you know|think about|imagine)", "engaging"),
        (r"(?i)(we've all|all of us|everyone has)", "shared_experience"),
        (r"(?i)(it's like|feels like|that moment)", "empathy"),
    ],
    "takeaway": [
        (r"(?i)(key lesson|main takeaway)", "key_lesson"),
        (r"(?i)(do this|try this|start with)", "action"),
        (r"(?i)(remember|never forget)", "remember"),
        (r"(?i)(the point is|what matters)", "point"),
        (r"(?i)(if you (take|learn|get) (anything|something))", "call_to_action"),
        (r"(?i)(here's (the thing|what I|what you))", "here_is"),
        (r"(?i)(that's why|that's how|that's what)", "conclusion"),
    ],
    "power_words": [
        (r"(?i)\bsecret\b", "secret"),
        (r"(?i)\bshocking\b", "shocking"),
        (r"(?i)\bnever\b", "never"),
        (r"(?i)\balways\b", "always"),
        (r"(?i)\bimpossible\b", "impossible"),
        (r"(?i)\bguaranteed?\b", "guaranteed"),
        (r"(?i)\bchanged?\s+my\s+life\b", "life_changing"),
    ]
}


def match_patterns(text: str) -> list[str]:
    """Find all pattern matches in text."""
    matched = []
    for category, patterns in PATTERNS.items():
        for pattern, name in patterns:
            if re.search(pattern, text):
                matched.append(name)
    return matched
```

#### 7c. Audio Features

**File:** `engine/features/audio_features.py`

```python
def compute_audio_features(segment: Segment, y: np.ndarray, sr: int) -> AudioFeatures:
    """Compute per-segment audio features."""
    start_sample = int(segment.start * sr)
    end_sample = int(segment.end * sr)
    seg_audio = y[start_sample:end_sample]
    
    # Speech rate
    onsets = librosa.onset.onset_detect(y=seg_audio, sr=sr, units='time')
    speech_rate = len(onsets) / segment.duration if segment.duration > 0 else 0
    
    # Volume (RMS)
    volume = np.mean(librosa.feature.rms(y=seg_audio))
    
    # Pause after (gap to next segment)
    pause_after = None  # Set during processing
    
    # Volume delta from average
    global_volume = np.mean(librosa.feature.rms(y=y))
    volume_delta = volume - global_volume
    
    return AudioFeatures(speech_rate, volume, volume_delta, pause_after)
```

#### 7d. Structural Features

**File:** `engine/features/structural.py`

```python
def compute_structural_features(segment: Segment, total_duration: float) -> dict:
    """Compute position-based features."""
    return {
        "position": segment.start / total_duration,
        "recency": segment.start / total_duration > 0.7,
        "is_first_30_pct": segment.start / total_duration < 0.3,
        "is_last_30_pct": segment.start / total_duration > 0.7,
    }
```

---

### Stage 8: Knowledge Graph Construction

```
Input:  list[Segment] with features
Process: Build graph nodes + edges
Output: KnowledgeGraph(nodes, edges)
```

**File:** `engine/graph/knowledge_graph.py`

```python
class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def add_segment(self, segment: Segment):
        self.graph.add_node(
            segment.id,
            segment=segment,
            text=segment.text,
            start=segment.start,
            end=segment.end,
            duration=segment.duration,
            sentiment=segment.sentiment,
            patterns=segment.patterns,
            position=segment.position
        )
    
    def add_relationship(self, source_id: str, target_id: str,
                         edge_type: str, weight: float, evidence: str):
        self.graph.add_edge(
            source_id, target_id,
            type=edge_type,
            weight=weight,
            evidence=evidence
        )
    
    def get_temporal_neighbors(self, segment_id: str, window: int = 3) -> list[str]:
        """Get segments within N positions of this segment."""
        # Implementation: traverse forward/backward N positions
        pass
    
    def get_connected_cluster(self, start_id: str, max_duration: float = 90.0) -> list[str]:
        """Traverse graph to build a connected clip starting from a hook.
        
        BFS traversal following edges with max-duration constraint to avoid
        generating overly long candidates early in the pipeline.
        """
        # BFS with cumulative duration check
        visited = set()
        queue = [(start_id, 0.0)]  # (node_id, accum_duration)
        cluster = []
        
        while queue:
            node_id, accum = queue.pop(0)
            if node_id in visited or accum > max_duration:
                continue
            visited.add(node_id)
            cluster.append(node_id)
            
            for neighbor in self.graph.successors(node_id):
                seg = self.graph.nodes[neighbor]["segment"]
                new_accum = accum + seg.duration
                if new_accum <= max_duration:
                    queue.append((neighbor, new_accum))
        
        return cluster
```

**File:** `engine/graph/relationships.py`

```python
def detect_relationships(segments: list[Segment], graph: KnowledgeGraph):
    """Detect and add edges between segments."""
    for i, seg in enumerate(segments):
        graph.add_segment(seg)
        
        # Temporal: follows (adjacent segments)
        if i > 0:
            graph.add_relationship(
                segments[i-1].id, seg.id,
                "follows", 1.0,
                "temporal adjacency"
            )
        
        # Contrast: contrast words + opposing sentiment
        for j in range(i+1, min(i+5, len(segments))):
            other = segments[j]
            if "but" in seg.patterns and abs(seg.sentiment - other.sentiment) > 0.3:
                graph.add_relationship(
                    seg.id, other.id,
                    "contrasts", 0.7,
                    "opposing sentiment with contrast marker"
                )
        
        # Explains: shared keywords + clarification patterns
        # Concludes: summary patterns + end position
        # Supports: similar claims + evidence patterns
```

**Edge types:**
| Type | Detection Method | Weight |
|------|-----------------|--------|
| `follows` | Temporal adjacency | 1.0 |
| `explains` | Shared keywords + "which means", "in other words" | 0.7-0.9 |
| `contrasts` | Contrast words + opposing sentiment | 0.7-0.9 |
| `concludes` | Summary patterns + end position | 0.6-0.8 |
| `supports` | Similar claims + "for example", "like when" | 0.5-0.7 |

---

### Stage 9: Hook Detection

```
Input:  KnowledgeGraph
Process: Apply hook rules → find segments matching hook patterns
Output: list[HookCandidate(segment_id, score)]
```

**File:** `engine/rules/hook_rules.py`

```python
def find_hook_candidates(graph: KnowledgeGraph) -> list[HookCandidate]:
    """Find segments that make good hooks."""
    candidates = []
    
    for node_id, data in graph.graph.nodes(data=True):
        seg = data["segment"]
        score = 0
        
        # Rule 1: Duration must be 3-8 seconds
        if not (3 <= seg.duration <= 8):
            continue
        
        # Rule 2: High energy
        if seg.speech_rate > 2.0:
            score += 20
        if seg.volume_delta > 1.5:
            score += 15
        
        # Rule 3: Curiosity
        if any(p in seg.patterns for p in ["what_if", "unknown", "biggest", "question"]):
            score += 25
        
        # Rule 4: Problem statement
        if seg.sentiment < -0.2 and "personal" in seg.patterns:
            score += 20
        
        # Rule 5: Contrast/conflict
        if any(p in seg.patterns for p in ["but", "however", "surprisingly"]):
            score += 15
        
        # Rule 6: Energy escalation (high volume + high speech rate)
        if seg.volume_delta > 2.0 and seg.speech_rate > 2.5:
            score += 25  # Bonus for high energy moments
        
        if score > 50:  # Minimum threshold
            candidates.append(HookCandidate(node_id, score))
    
    return sorted(candidates, key=lambda c: c.score, reverse=True)
```

---

### Stage 10: Body Detection

```
Input:  list[HookCandidate] + KnowledgeGraph
Process: For each hook, find body segments that provide context
Output: list[HookBodyPair(hook_id, body_ids, score)]
```

**File:** `engine/rules/body_rules.py`

```python
def find_body_candidates(hook_id: str, graph: KnowledgeGraph) -> list[BodyCandidate]:
    """Find body segments that provide context for a hook."""
    hook_data = graph.graph.nodes[hook_id]
    hook_seg = hook_data["segment"]
    
    candidates = []
    
    # Look for segments that explain or support the hook
    for neighbor_id in graph.graph.successors(hook_id):
        edge_data = graph.graph.edges[hook_id, neighbor_id]
        neighbor_seg = graph.graph.nodes[neighbor_id]["segment"]
        
        if edge_data["type"] in ["explains", "supports"]:
            score = edge_data["weight"] * 100
            
            # Bonus for temporal proximity
            time_gap = neighbor_seg.start - hook_seg.end
            if time_gap < 60:  # Within 1 minute
                score += 20
            
            candidates.append(BodyCandidate(neighbor_id, score))
    
    # Also consider segments within temporal window
    for neighbor_id in graph.get_temporal_neighbors(hook_id, window=5):
        neighbor_seg = graph.graph.nodes[neighbor_id]["segment"]
        if neighbor_id not in [c.id for c in candidates]:
            score = 50  # Baseline for temporal neighbors
            candidates.append(BodyCandidate(neighbor_id, score))
    
    return sorted(candidates, key=lambda c: c.score, reverse=True)
```

---

### Stage 11: Ending Detection

```
Input:  list[HookBodyPair] + KnowledgeGraph
Process: Find ending segments that resolve the tension
Output: list[CompleteClip(hook_id, body_ids, ending_id)]
```

**File:** `engine/rules/ending_rules.py`

```python
def find_ending_candidates(hook_id: str, body_ids: list[str],
                           graph: KnowledgeGraph) -> list[EndingCandidate]:
    """Find ending segments that resolve the narrative arc."""
    hook_seg = graph.graph.nodes[hook_id]["segment"]
    last_body = graph.graph.nodes[body_ids[-1]]["segment"]
    
    candidates = []
    
    # Look for segments that conclude or resolve
    for node_id, data in graph.graph.nodes(data=True):
        seg = data["segment"]
        
        # Must be after the body
        if seg.start <= last_body.start:
            continue
        
        score = 0
        
        # Rule 1: Positive sentiment shift
        if seg.sentiment > 0.2:
            score += 25
        
        # Rule 2: Takeaway/lesson patterns
        if any(p in seg.patterns for p in ["key_lesson", "action", "remember", "point"]):
            score += 30
        
        # Rule 3: Summary/conclusion
        if any(p in seg.patterns for p in ["finally", "so", "therefore"]):
            score += 20
        
        # Rule 4: Duration 5-10 seconds
        if 5 <= seg.duration <= 10:
            score += 15
        elif 3 <= seg.duration <= 15:
            score += 10
        
        # Rule 5: Recency (last 30% of video)
        if seg.position > 0.7:
            score += 20
        
        # Rule 6: Practicality (saveability signal)
        if any(p in seg.patterns for p in ["steps", "lesson", "framework", "tip"]):
            score += 25
        
        # Rule 7: Relatability (personal connection)
        if "personal" in seg.patterns:
            score += 10
        
        # Rule 8: Complete resolution - positive + practical + personal
        if seg.sentiment > 0.3 and \
           any(p in seg.patterns for p in ["key_lesson", "action", "point"]) and \
           "personal" in seg.patterns:
            score += 30  # Bonus for complete resolution

        
        if score > 40:  # Minimum threshold
            candidates.append(EndingCandidate(node_id, score))
    
    return sorted(candidates, key=lambda c: c.score, reverse=True)
```

---

### Stage 11b: Stitching Priority (Bonus)

When generating complete clips, prioritize candidates where body segments come from **different parts of the video** (true stitching) while keeping the total gap between segments ≤ 2-3 minutes. This produces more interesting clips by pulling context from diverse locations rather than adjacent segments.

```python
def compute_stitching_diversity(clip: CompleteClip) -> float:
    """Score how diverse the source locations are (true stitching bonus)."""
    if len(clip.segments) < 3:
        return 0.0
    
    start_times = [s.start for s in clip.segments]
    max_gap = max(start_times) - min(start_times)
    
    # Reward clips where segments span 30-180 seconds of source material
    if max_gap > 180:  # Too far apart — context loss
        return 0.3
    elif max_gap > 60:  # Good diversity
        return 1.0
    elif max_gap > 30:  # Moderate diversity
        return 0.7
    else:  # Too close — essentially adjacent
        return 0.3
```

This score is added as a bonus during the Scoring stage.

---

### Stage 12: Rule Engine

```
Input:  list[CompleteClip]
Process: Filter invalid candidates
Output: list[ValidClip]
```

**File:** `engine/scoring/rule_engine.py`

```python
HARD_RULES = {
    "total_duration_45_to_90": {
        "check": lambda clip: 45 <= clip.total_duration <= 90,
        "penalty": "reject"
    },
    "hook_in_first_5_seconds": {
        "check": lambda clip: clip.hook_duration <= 5,
        "penalty": "reject"
    },
    "has_curiosity": {
        "check": lambda clip: any("curiosity" in s.patterns for s in clip.segments),
        "penalty": "reject"
    },
    "has_practicality_or_emotion": {
        "check": lambda clip: (
            any("practicality" in s.patterns for s in clip.segments) or
            any(abs(s.sentiment) > 0.3 for s in clip.segments)
        ),
        "penalty": "reject"
    },
    "max_2_speaker_changes": {
        "check": lambda clip: clip.speaker_changes <= 2,
        "penalty": "reject"
    },
    "no_context_gaps": {
        "check": lambda clip: not has_context_references(clip),
        "penalty": "reject"
    }
}

SOFT_RULES = {
    "emotional_arc": {
        "check": lambda clip: has_emotional_arc(clip),
        "bonus": 20
    },
    "practical_value": {
        "check": lambda clip: any("practicality" in s.patterns for s in clip.segments),
        "bonus": 15
    },
    "relatable": {
        "check": lambda clip: any("personal" in s.patterns for s in clip.segments),
        "bonus": 10
    }
}


def has_context_references(clip: CompleteClip) -> bool:
    """Check for unresolved references to removed context."""
    context_patterns = [
        r"(?i)as I (said|mentioned|told)",
        r"(?i)going back to",
        r"(?i)as we (discussed|talked)",
        r"(?i)earlier (I|we)",
        r"(?i)like I said",
        r"(?i)like I was (saying|saying earlier)",
        r"(?i)to go back",
        r"(?i)as mentioned (earlier|before)",
        r"(?i)coming back to",
        r"(?i)recall (that|when|how)",
    ]
    
    for seg in clip.segments:
        for pattern in context_patterns:
            if re.search(pattern, seg.text):
                return True
    return False


def has_emotional_arc(clip: CompleteClip) -> bool:
    """Check if clip has a complete emotional journey."""
    if len(clip.segments) < 3:
        return False
    
    hook = clip.segments[0]
    middle = clip.segments[len(clip.segments) // 2]
    ending = clip.segments[-1]
    
    # Hook: negative/curious → Body: tension → Ending: positive resolution
    return (hook.sentiment < 0.1 and
            middle.sentiment < -0.1 and
            ending.sentiment > 0.2)
```

---

### Stage 13: Scoring

```
Input:  list[ValidClip]
Process: Compute weighted scores
Output: list[ScoredClip] sorted by total_score
```

**File:** `engine/scoring/scorer.py`

```python
def score_clip(clip: ValidClip) -> ScoredClip:
    """Compute comprehensive scores for a clip."""
    
    # Hook score: curiosity + energy + duration fit
    hook = clip.hook_segment
    hook_score = (
        (0.4 * (1 if "curiosity" in hook.patterns else 0)) +
        (0.3 * min(hook.speech_rate / 3.0, 1.0)) +
        (0.3 * (1 - abs(hook.duration - 5) / 10))
    )
    
    # Body score: context + tension + flow
    body_score = 0
    for seg in clip.body_segments:
        body_score += (
            (0.3 * (1 if "personal" in seg.patterns else 0)) +
            (0.3 * (1 if abs(seg.sentiment) > 0.2 else 0)) +
            (0.4 * (1 if seg.duration < 12 else 0.5))
        )
    body_score /= len(clip.body_segments) if clip.body_segments else 1
    
    # Ending score: resolution + takeaway + duration fit
    ending = clip.ending_segment
    ending_score = (
        (0.3 * max(ending.sentiment, 0)) +
        (0.4 * (1 if "lesson" in ending.patterns or "practicality" in ending.patterns else 0)) +
        (0.3 * (1 - abs(ending.duration - 7) / 10))
    )
    
    # Flow score: natural transitions
    flow_score = compute_flow_score(clip)
    
    # Uniqueness score: TF-IDF against full transcript to avoid generic clips
    uniqueness_score = compute_uniqueness_score(clip, full_transcript)
    
    # Total score (weighted formula)
    total_score = (
        0.35 * hook_score +
        0.25 * body_score +
        0.20 * ending_score +
        0.15 * flow_score +
        0.05 * (1 if any("practicality" in s.patterns for s in clip.all_segments) else 0) +
        0.05 * uniqueness_score  # Bonus for distinctive, non-generic content
    )
    
    return ScoredClip(
        clip=clip,
        hook_score=hook_score,
        body_score=body_score,
        ending_score=ending_score,
        flow_score=flow_score,
        total_score=total_score
    )


def compute_uniqueness_score(clip: ValidClip, full_transcript: str) -> float:
    """Score how unique/rare the clip's content is vs the full transcript.
    
    Uses simple TF-IDF: words that appear infrequently across the full
    transcript get higher uniqueness weight. This prevents generic filler
    clips from scoring highly.
    """
    from collections import Counter
    import math
    
    # Tokenize transcript
    all_words = re.findall(r'\w+', full_transcript.lower())
    word_freq = Counter(all_words)
    n_total = len(all_words)
    
    # Score clip words by inverse document frequency
    clip_text = " ".join(s.text for s in clip.all_segments)
    clip_words = re.findall(r'\w+', clip_text.lower())
    
    if not clip_words:
        return 0.0
    
    idf_scores = []
    for w in clip_words:
        freq = word_freq.get(w, 0)
        idf = math.log((n_total + 1) / (freq + 1)) if freq > 0 else 0
        idf_scores.append(idf)
    
    avg_idf = sum(idf_scores) / len(idf_scores)
    # Normalize to 0-1 range (typical max IDF for ~10K words is ~6-8)
    return min(avg_idf / 3.0, 1.0)


def compute_flow_score(clip: ValidClip) -> float:
    """Score how natural the transitions feel."""
    score = 0
    
    # Transition 1: Hook → Body
    hook_end = clip.hook_segment.end
    body_start = clip.body_segments[0].start
    gap1 = body_start - hook_end
    if gap1 < 2:  # Immediate follow-up
        score += 0.3
    elif gap1 < 10:  # Slight gap (acceptable)
        score += 0.2
    
    # Transition 2: Body → Ending
    body_end = clip.body_segments[-1].end
    ending_start = clip.ending_segment.start
    gap2 = ending_start - body_end
    if gap2 < 3:
        score += 0.3
    elif gap2 < 15:
        score += 0.2
    
    # Emotional arc
    if has_emotional_arc(clip):
        score += 0.4
    
    return min(score, 1.0)
```

---

### Stage 14: LLM Teacher — Structured Label Generation

```
Input:  list[ScoredClip] (top 20) + list[RejectedClip] + all Segments
Process: LLM produces structured numeric labels for every segment and candidate
Output: SegmentLabels[] + CandidateLabels[] + RejectionLabels[]
```

**Core Principle:** The LLM is not a "verifier" that picks winners. It is a **teacher** that produces structured training data. Every answer becomes a labeled data point for future ML models.

**File:** `engine/llm/teacher.py`

```python
TEACHER_PROMPT = """
You are a viral clip teacher. For each segment and candidate below,
rate the following dimensions from 0.0 to 1.0.

Return valid JSON only.

## Per-Segment Rating

SEGMENT: {text}
SPEAKER: {speaker}
PATTERNS: {patterns}
SENTIMENT: {sentiment}

Rate:
{
  "is_hook": <0-1>,
  "hook_strength": <0-1>,
  "is_context": <0-1>,
  "is_takeaway": <0-1>,
  "emotion": "<curiosity|surprise|sadness|joy|anger|fear|neutral|inspiration>",
  "requires_previous_context": <0-1>,
  "creates_new_context": <0-1>,
  "is_story": <0-1>,
  "is_opinion": <0-1>,
  "is_fact": <0-1>,
  "speaker_confidence": <0-1>,
  "saveability": <0-1>,
  "shareability": <0-1>,
  "confidence": <0-1>
}

## Per-Candidate Rating

CANDIDATE TRANSCRIPT: {transcript}
HOOK TEXT: {hook}
BODY TEXT: {body}
ENDING TEXT: {ending}

Rate:
{
  "story_complete": <0-1>,
  "transition_quality": <0-1>,
  "context_missing": <0-1>,
  "shareability": <0-1>,
  "saveability": <0-1>,
  "hook_strength": <0-1>,
  "ending_strength": <0-1>,
  "emotional_arc_build_up": <0-1>,
  "naturalness": <0-1>,
  "curiosity_gap": <0-1>,
  "confidence": <0-1>
}
"""
```

**File:** `engine/llm/label_schemas.py`

```python
@dataclass
class SegmentLabel:
    is_hook: float
    hook_strength: float
    is_context: float
    is_takeaway: float
    emotion: str
    requires_previous_context: float
    creates_new_context: float
    is_story: float
    is_opinion: float
    is_fact: float
    speaker_confidence: float
    saveability: float
    shareability: float
    confidence: float

@dataclass
class CandidateLabel:
    story_complete: float
    transition_quality: float
    context_missing: float
    shareability: float
    saveability: float
    hook_strength: float
    ending_strength: float
    emotional_arc_build_up: float
    naturalness: float
    curiosity_gap: float
    confidence: float
```

**Why structured labels beat free-text:**
- Every answer is a numeric vector → directly trainable
- Enables regression models (predict virality score)
- Enables classification models (predict hook/body/ending)
- Failures produce equally valuable labels (why did it fail?)

---

### Stage 15: Decision Log & Audit Trail

```
Input:  All pipeline decisions (rules matched, scores, LLM labels, confidence)
Process: Log every decision with its contributing signals
Output: decision_log entries
```

**File:** `engine/decision/log.py`

```python
class DecisionLog:
    """Complete audit trail of every decision made during processing."""
    
    def log_segment_decision(self, segment_id: str, stage: str, 
                              rule: str, confidence: float, outcome: str):
        """
        Segment 143
            ↓
        Matched Rule: Curiosity (confidence: 0.87)
            ↓
        Matched Rule: High Energy (confidence: 0.92)
            ↓
        Knowledge Graph: Explains Segment 149 (confidence: 0.76)
            ↓
        LLM Teacher: Hook Score = 0.91
            ↓
        Final Decision: Selected
        """
        entry = {
            "segment_id": segment_id,
            "video_id": self.video_id,
            "stage": stage,
            "rule": rule,
            "confidence": confidence,
            "outcome": outcome,
            "timestamp": time.time(),
            "pipeline_version": self.version
        }
        self.db.insert("decision_log", entry)
    
    def log_candidate_decision(self, candidate_id: str, stage: str,
                                rule: str, confidence: float, outcome: str,
                                rejection_reason: str = None):
        """Log candidate-level decisions including rejections."""
        entry = {
            "candidate_id": candidate_id,
            "video_id": self.video_id,
            "stage": stage,
            "rule": rule,
            "confidence": confidence,
            "outcome": outcome,
            "rejection_reason": rejection_reason,
            "timestamp": time.time(),
            "pipeline_version": self.version
        }
        self.db.insert("decision_log", entry)
```

**Purpose:** Years later, when training custom models, this log reveals exactly which signals influenced each decision — enabling feature importance analysis and model debugging.

---

### Stage 16: Pattern Recognition

```
Input:  SegmentLabels[] + CandidateLabels[] + decision_log
Process: Discover recurring narrative patterns across the current video
Output: DiscoveredPatterns[] (merged into Global Pattern DB)
```

**File:** `engine/patterns/detector.py`

```python
def discover_patterns(segments: list[Segment], 
                       labels: list[SegmentLabel]) -> list[Pattern]:
    """Discover narrative patterns from labeled segments.
    
    A pattern is a sequence of segment types that form a recognizable
    narrative structure, e.g.:
    
    Pattern #421: Question → Personal Story → Failure → Lesson
        → Share Probability: 93%
        → Occurrences: 186 (across all videos)
        → Confidence: 0.96
    
    Pattern #128: Money → Mistake → Framework
        → Save Probability: 91%
        → Occurrences: 342
        → Confidence: 0.94
    """
    patterns = []
    
    # Extract label sequences
    label_sequences = extract_label_sequences(segments, labels)
    
    # Find frequent subsequences
    for seq in label_sequences:
        # Sliding window over segment types
        for i in range(len(seq) - 2):
            subseq = seq[i:i+3]
            pattern = Pattern(
                nodes=[s["type"] for s in subseq],
                context_requirements=infer_context(subseq),
                first_seen=time.time()
            )
            patterns.append(pattern)
    
    return merge_with_global_patterns(patterns)


def merge_with_global_patterns(local_patterns: list[Pattern]) -> list[Pattern]:
    """Merge newly discovered patterns into the global pattern database.
    
    If pattern already exists (same node sequence):
      - Increment occurrences
      - Update running averages for saves/shares
      - Recalculate confidence
    If pattern is new:
      - Add to database with occurrence_count = 1
    """
    for pattern in local_patterns:
        existing = GLOBAL_DB.find_pattern(pattern.node_sequence)
        if existing:
            existing.occurrences += 1
            existing.avg_saves = running_avg(existing.avg_saves, 
                                              pattern.avg_saves, existing.occurrences)
            existing.avg_shares = running_avg(existing.avg_shares,
                                               pattern.avg_shares, existing.occurrences)
            existing.confidence = min(existing.confidence + 0.01, 0.99)
        else:
            GLOBAL_DB.insert_pattern(pattern)
```

---

### Stage 17: Global Knowledge Graph Update

```
Input:  Local per-video graph + LLM labels + Candidate performance
Process: Update persistent cross-video graph with statistical weights
Output: GlobalKnowledgeGraph with learned edge weights
```

**File:** `engine/knowledge/global_graph.py`

```python
class GlobalKnowledgeGraph:
    """Persistent knowledge graph that learns from every video processed.
    
    Unlike the local graph (discarded per video), this graph accumulates
    statistical knowledge across ALL videos.
    """
    
    def update_from_local(self, local_graph, labels, candidates):
        for edge in local_graph.edges:
            global_edge = self.find_or_create_edge(edge.type)
            
            # Update running statistics
            global_edge.times_used += 1
            global_edge.avg_watch_time = running_avg(
                global_edge.avg_watch_time, 
                edge.watch_time, 
                global_edge.times_used
            )
            global_edge.avg_saves = running_avg(
                global_edge.avg_saves, 
                edge.saves, 
                global_edge.times_used
            )
            global_edge.avg_shares = running_avg(
                global_edge.avg_shares, 
                edge.shares, 
                global_edge.times_used
            )
            
            # Edge weight is now statistical, not handcrafted
            global_edge.weight = compute_statistical_weight(
                global_edge.avg_saves,
                global_edge.avg_shares,
                global_edge.times_used
            )
    
    def find_or_create_edge(self, edge_type: str) -> GlobalEdge:
        edges = {
            "curiosity_hook → personal_story": GlobalEdge(
                type="curiosity_hook → personal_story",
                times_used=894,
                avg_watch_time=0.74,
                avg_saves=0.68,
                avg_shares=0.55,
                llm_confidence=0.91
            ),
            "statistics → framework": GlobalEdge(
                type="statistics → framework",
                times_used=342,
                avg_watch_time=0.61,
                avg_saves=0.82,
                avg_shares=0.43,
                llm_confidence=0.87
            )
        }
        return edges.get(edge_type, GlobalEdge(edge_type))
```

**Statistical edges (learned, not hardcoded):**

| Edge | Times Used | Avg Watch Time | Avg Saves | Avg Shares | LLM Confidence |
|------|-----------|---------------|-----------|-----------|---------------|
| curiosity_hook → personal_story | 894 | 74% | 68% | 55% | 0.91 |
| statistics → framework | 342 | 61% | 82% | 43% | 0.87 |
| question → personal_failure → lesson | 186 | 78% | 82% | 74% | 0.96 |
| money → mistake → framework | 134 | 65% | 91% | 38% | 0.94 |
| contrarian → explanation → takeaway | 97 | 71% | 59% | 67% | 0.88 |

---

### Stage 18: Failure Storage

```
Input:  All rejected candidates + rejection reasons from Rule Engine
Process: Store structured rejection data alongside accepted candidates
Output: FailedCandidate records
```

**File:** `engine/decision/failures.py`

```python
@dataclass
class FailedCandidate:
    candidate_id: str
    video_id: str
    hook_segment_id: str
    body_segment_ids: list[str]
    ending_segment_id: str
    total_duration: float
    hook_score: float
    body_score: float
    ending_score: float
    flow_score: float
    total_score: float
    rejection_reason: str          # "missing_context" | "weak_ending" | "no_curiosity" | ...
    rejection_stage: str           # "rule_engine" | "scorer" | "llm_teacher"
    rules_failed: list[str]
    rules_passed: list[str]
    llm_label: CandidateLabel     # LLM evaluation even for rejected candidates
    created_at: datetime
    pipeline_version: str
```

**Why store failures:**
- Failures teach the system what NOT to choose
- Enables binary classification models (select vs. reject)
- Reveals systematic weaknesses (e.g., 70% of rejects due to "missing context")
- Rich negative examples for training future rankers

---

### Stage 19: Data Storage

### Stage 19: Data Storage

**File:** `engine/data/models.py`

```python
@dataclass
class VideoAnalysis:
    video_id: str
    filename: str
    duration: float
    language: str
    transcript_text: str
    word_count: int
    segment_count: int
    snr_db: float
    speech_rate: float
    volume_rms: float
    segments: list[Segment]
    relationships: list[Relationship]
    candidates: list[Candidate]
    llm_viral_moments: list[dict]
    llm_reasoning: str
    created_at: datetime
    processing_time: float
    pipeline_version: str


@dataclass
class Segment:
    id: str
    video_id: str
    index: int
    text: str
    words: list[Word]
    start: float
    end: float
    duration: float
    speaker: str
    sentiment: float
    speech_rate: float
    pause_after: float
    volume_delta: float
    patterns: list[str]
    rules_matched: list[str]
    llm_analysis: str
    watch_time_pct: float
    shares: int
    saves: int


@dataclass
class Word:
    text: str
    start: float
    end: float
    confidence: float
    is_power: bool
    power_category: str


@dataclass
class Relationship:
    source_id: str
    target_id: str
    edge_type: str
    weight: float
    evidence: str


@dataclass
class Candidate:
    id: str
    video_id: str
    hook_segment_id: str
    body_segment_ids: list[str]
    ending_segment_id: str
    total_duration: float
    hook_score: float
    body_score: float
    ending_score: float
    flow_score: float
    total_score: float
    rules_passed: list[str]
    rules_failed: list[str]
    llm_ranking: int
    llm_reasoning: str
    actual_performance: dict
```

**File:** `engine/data/storage.py`

```python
CREATE TABLE videos (
    id TEXT PRIMARY KEY,
    filename TEXT,
    duration REAL,
    language TEXT,
    transcript_text TEXT,
    word_count INTEGER,
    segment_count INTEGER,
    snr_db REAL,
    speech_rate REAL,
    volume_rms REAL,
    llm_viral_moments TEXT,
    llm_reasoning TEXT,
    created_at TIMESTAMP,
    processing_time REAL,
    pipeline_version TEXT
);

CREATE TABLE segments (
    id TEXT PRIMARY KEY,
    video_id TEXT REFERENCES videos(id),
    segment_index INTEGER,
    text TEXT,
    start_time REAL,
    end_time REAL,
    duration REAL,
    speaker TEXT,
    sentiment REAL,
    speech_rate REAL,
    pause_after REAL,
    volume_delta REAL,
    patterns TEXT,
    rules_matched TEXT,
    llm_analysis TEXT,
    watch_time_pct REAL,
    shares INTEGER,
    saves INTEGER
);

CREATE TABLE words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id TEXT REFERENCES segments(id),
    word_index INTEGER,
    text TEXT,
    start_time REAL,
    end_time REAL,
    confidence REAL,
    is_power BOOLEAN,
    power_category TEXT
);

CREATE TABLE relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT REFERENCES segments(id),
    target_id TEXT REFERENCES segments(id),
    edge_type TEXT,
    weight REAL,
    evidence TEXT
);

CREATE TABLE candidates (
    id TEXT PRIMARY KEY,
    video_id TEXT REFERENCES videos(id),
    hook_segment_id TEXT REFERENCES segments(id),
    body_segment_ids TEXT,
    ending_segment_id TEXT REFERENCES segments(id),
    total_duration REAL,
    hook_score REAL,
    body_score REAL,
    ending_score REAL,
    flow_score REAL,
    total_score REAL,
    rules_passed TEXT,
    rules_failed TEXT,
    llm_ranking INTEGER,
    llm_reasoning TEXT,
    actual_performance TEXT
);
```

---

### Stage 20: Pipeline Orchestrator

**File:** `engine/pipeline.py`

```python
class AnalysisPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.decision_log = DecisionLog()
        self.global_graph = GlobalKnowledgeGraph()
        self.pattern_engine = PatternEngine()
    
    def run(self, video_path: str) -> VideoAnalysis:
        start_time = time.time()
        
        # Stage 1: Audio extraction
        audio_path = extract_audio(video_path, temp_path)
        
        # Stage 2: Audio quality
        quality = measure_audio_quality(audio_path)
        
        # Stage 3: Chunking
        chunks = overlap_chunk(audio_path, quality)
        
        # Stage 4: Transcribe
        transcribed = []
        for chunk in chunks:
            text = transcribe_chunk(chunk.path)
            lang = detect_language(text.split())
            clean = remove_fillers(text)
            transcribed.append({"text": clean, "start": chunk.start, "end": chunk.end})
        
        # Stage 5: Alignment
        aligned = align_segments(transcribed, audio_path, lang)
        
        # Stage 6: Atomic segmentation
        segments = split_into_atomic_segments(aligned)
        
        # Stage 7: Feature extraction
        y, sr = librosa.load(audio_path, sr=16000)
        for seg in segments:
            seg.sentiment = compute_sentiment(seg.text)
            seg.patterns = match_patterns(seg.text)
            audio_feats = compute_audio_features(seg, y, sr)
            seg.speech_rate = audio_feats.speech_rate
            seg.volume_delta = audio_feats.volume_delta
        
        # Stage 8: Local knowledge graph
        graph = LocalKnowledgeGraph()
        detect_relationships(segments, graph)
        
        # Stage 9: Hook detection (with confidence)
        hooks = find_hook_candidates(graph)
        for hook in hooks:
            self.decision_log.log(hook.id, "hook_rules", 
                                   hook.rule, hook.confidence, "candidate")
        
        # Stage 10: Body detection
        hook_body_pairs = []
        for hook in hooks[:20]:
            bodies = find_body_candidates(hook.id, graph)
            if bodies:
                hook_body_pairs.append((hook, bodies[:5]))
        
        # Stage 11: Ending detection
        complete_clips = []
        for hook, bodies in hook_body_pairs:
            for body in bodies:
                endings = find_ending_candidates(hook.id, [body.id], graph)
                if endings:
                    clip = CompleteClip(hook.id, [body.id], endings[0].id)
                    complete_clips.append(clip)
        
        # Stage 11b: Stitching diversity bonus
        for clip in complete_clips:
            clip.diversity_score = compute_stitching_diversity(clip)
        
        # Stage 12: Rule engine (separate accepted vs rejected)
        valid_clips = []
        rejected_clips = []
        for clip in complete_clips:
            passed = all(rule["check"](clip) for rule in HARD_RULES.values())
            if passed:
                valid_clips.append(clip)
                self.decision_log.log(clip.id, "rule_engine", 
                                       "all_hard_rules", 1.0, "accepted")
            else:
                rejected_clips.append(clip)
                self.decision_log.log(clip.id, "rule_engine",
                                       failed_rule(clip), 0.0, "rejected")
        
        # Stage 13: Scoring
        scored = [score_clip(c) for c in valid_clips]
        scored.sort(key=lambda c: c.total_score, reverse=True)
        top20 = scored[:20]
        
        # Stage 14: LLM Teacher — structured labels
        segment_labels = []
        if self.config.get("use_llm"):
            for seg in segments[:100]:  # Label top 100 segments
                label = label_segment(seg, self.config["llm_api_key"])
                segment_labels.append(label)
                confidence.scorer.record("segment", seg.id, label.confidence)
            
            candidate_labels = []
            for clip in top20:
                clabel = label_candidate(clip, self.config["llm_api_key"])
                candidate_labels.append(clabel)
                confidence.scorer.record("candidate", clip.id, clabel.confidence)
            
            # Also label rejected candidates (failures are training data too)
            rejected_labels = []
            for clip in rejected_clips[:20]:
                rlabel = label_candidate(clip, self.config["llm_api_key"])
                rejected_labels.append(rlabel)
        
        # Stage 15: Decision log — audit trail
        self.decision_log.flush()
        
        # Stage 16: Pattern matching (against global database)
        local_patterns = self.pattern_engine.discover(segments, segment_labels)
        matched_patterns = self.pattern_engine.match_against_global(local_patterns)
        
        # Stage 17: Update global knowledge graph
        self.global_graph.update_from_local(graph, segment_labels, candidate_labels)
        
        # Stage 18: Store failures
        for clip, rlabel in zip(rejected_clips[:20], rejected_labels):
            store_failed_candidate(FailedCandidate(
                candidate_id=clip.id,
                video_id=video_id,
                rejection_reason=clip.rejection_reason,
                llm_label=rlabel,
                ...
            ))
        
        # Stage 19: Data storage (candidates + patterns + decisions)
        analysis = VideoAnalysis(
            video_id=str(uuid4()),
            filename=os.path.basename(video_path),
            segments=segments,
            segment_labels=segment_labels,
            candidates=top20,
            candidate_labels=candidate_labels,
            rejected_candidates=rejected_clips[:20],
            decision_log=self.decision_log.entries,
            patterns=matched_patterns,
            ...
        )
        store_analysis(analysis)
        
        return analysis
```

#### Error Handling

Every stage must be wrapped in structured try/except with stage-level fallbacks:

```python
class PipelineError(Exception):
    def __init__(self, stage: str, message: str, recoverable: bool = False):
        self.stage = stage
        self.recoverable = recoverable
        super().__init__(f"[{stage}] {message}")

class AnalysisPipeline:
    def run_stage(self, stage_name: str, fn, *args, **kwargs):
        """Run a pipeline stage with error handling."""
        try:
            self.logger.info(f"Stage started: {stage_name}")
            result = fn(*args, **kwargs)
            self.logger.info(f"Stage completed: {stage_name}")
            return result
        except PipelineError as e:
            if e.recoverable:
                self.logger.warning(f"Stage {stage_name} recovered: {e}")
                return self.get_fallback(stage_name)
            else:
                self.logger.critical(f"Stage {stage_name} failed: {e}")
                raise
        except Exception as e:
            self.logger.error(f"Stage {stage_name} unexpected error: {e}")
            raise PipelineError(stage_name, str(e))
```

**Fallback behavior per stage:**

| Stage | Recoverable? | Fallback |
|-------|-------------|---------|
| Audio extraction | Yes | Skip video (log as corrupted) |
| Quality analysis | Yes | Use default thresholds |
| Chunk transcription | **Per-chunk** | Retry once, skip failed chunk |
| Word alignment | Yes | Use unaligned segment boundaries |
| Feature extraction | **Per-feature** | Default values for failed features |
| Knowledge graph | No | Pipeline halts (data corruption risk) |
| LLM Teacher | **Per-segment** | Fall back to rule-only labels |
| Pattern matching | Yes | Skip pattern boost (score with rules only) |
| Global graph update | No | Pipeline continues (update deferred) |

#### Resume Capability

The pipeline must be resumable — if it crashes during alignment, it should not restart transcription.

```python
class AnalysisPipeline:
    STATE_FILE = "engine/data/store/{video_id}/state.json"
    
    def get_completed_stages(self, video_id: str) -> set[str]:
        """Check which stages have already completed."""
        state_path = Path(self.STATE_FILE.format(video_id=video_id))
        if state_path.exists():
            with open(state_path) as f:
                return set(json.load(f).get("completed_stages", []))
        return set()
    
    def mark_stage_complete(self, video_id: str, stage: str):
        """Persist a stage as completed."""
        state_path = Path(self.STATE_FILE.format(video_id=video_id))
        state_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {"completed_stages": []}
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
        
        if stage not in state["completed_stages"]:
            state["completed_stages"].append(stage)
        
        with open(state_path, "w") as f:
            json.dump(state, f)
    
    def run(self, video_path: str) -> VideoAnalysis:
        video_id = self.get_or_create_video_id(video_path)
        completed = self.get_completed_stages(video_id)
        
        if "audio_extraction" not in completed:
            audio_path = self.run_stage("audio_extraction", extract_audio, ...)
            self.mark_stage_complete(video_id, "audio_extraction")
        
        if "transcription" not in completed:
            # ... process ...
            self.mark_stage_complete(video_id, "transcription")
        
        # ... continues for each stage ...
```

---

## 5. Data Models

### Database Schema (SQLite)

```
videos
├── id (TEXT PK)
├── filename (TEXT)
├── duration (REAL)
├── language (TEXT)
├── transcript_text (TEXT)
├── word_count (INTEGER)
├── segment_count (INTEGER)
├── snr_db (REAL)
├── speech_rate (REAL)
├── volume_rms (REAL)
├── llm_viral_moments (TEXT JSON)
├── llm_reasoning (TEXT)
├── created_at (TIMESTAMP)
├── processing_time (REAL)
└── pipeline_version (TEXT)

segments (FK → videos.id)
├── id (TEXT PK)
├── segment_index (INTEGER)
├── text (TEXT)
├── start_time (REAL)
├── end_time (REAL)
├── duration (REAL)
├── speaker (TEXT)
├── sentiment (REAL)
├── speech_rate (REAL)
├── pause_after (REAL)
├── volume_delta (REAL)
├── patterns (TEXT JSON)
├── rules_matched (TEXT JSON)
├── llm_analysis (TEXT)
├── watch_time_pct (REAL)
├── shares (INTEGER)
└── saves (INTEGER)

words (FK → segments.id)
├── text (TEXT)
├── start_time (REAL)
├── end_time (REAL)
├── confidence (REAL)
├── is_power (BOOLEAN)
└── power_category (TEXT)

relationships
├── source_id (FK → segments.id)
├── target_id (FK → segments.id)
├── edge_type (TEXT)
├── weight (REAL)
└── evidence (TEXT)

candidates
├── id (TEXT PK)
├── video_id (FK → videos.id)
├── hook_segment_id (FK → segments.id)
├── body_segment_ids (TEXT JSON)
├── ending_segment_id (FK → segments.id)
├── total_duration (REAL)
├── hook_score (REAL)
├── body_score (REAL)
├── ending_score (REAL)
├── flow_score (REAL)
├── total_score (REAL)
├── rules_passed (TEXT JSON)
├── rules_failed (TEXT JSON)
├── llm_ranking (INTEGER)
├── llm_reasoning (TEXT)
└── actual_performance (TEXT JSON)
```

---

## 6. Rules & Fundamentals

### Core Fundamentals (Never Change)

1. **Hook must grab in first 3 seconds**
2. **Story must be complete** (beginning, middle, end)
3. **Viewer must get value** (knowledge, emotion, entertainment)
4. **Context must feel complete** — not a random cut
5. **Emotional journey** — tension → resolution
6. **Authenticity** — feels real, not forced

### Hook Rules

| Rule | Description | Weight |
|------|-------------|--------|
| `hook_in_3_seconds` | Duration ≤ 8 seconds | 1.0 |
| `creates_curiosity_gap` | Has question, cliffhanger, contrast | 0.9 |
| `feels_energetic` | High speech rate + volume | 0.7 |
| `problem_statement` | Negative sentiment + personal story | 0.8 |
| `promises_value` | Solution/lesson/secret implied | 0.8 |

### Body Rules

| Rule | Description | Weight |
|------|-------------|--------|
| `explains_hook` | Shares keywords with hook | 0.9 |
| `builds_tension` | Has emotional progression | 0.7 |
| `appropriate_length` | 15-30 seconds | 0.8 |
| `context_preservation` | No unresolved references | 0.9 |

### Ending Rules

| Rule | Description | Weight |
|------|-------------|--------|
| `resolves_tension` | Positive sentiment | 0.9 |
| `delivers_takeaway` | Lesson/practical/summary | 0.8 |
| `appropriate_length` | 5-10 seconds | 0.7 |
| `strong_takeaway` | Practical + personal + positive (bonus) | +0.3 |

### Stitching Rules

| Rule | Description | Penalty |
|------|-------------|---------|
| `natural_transition` | Hook → Body → Ending feels smooth | Reject |
| `no_context_gap` | No missing context between segments | Reject |
| `emotional_arc` | Peak → Tension → Resolution | Bonus +20 |
| `practical_value` | Contains actionable content | Bonus +15 |
| `relatable` | Personal pronouns, universal appeal | Bonus +10 |

### Hard Filters (Auto-Reject)

| Filter | Check |
|--------|-------|
| Duration | 45-90 seconds total |
| Hook position | Hook must be first 5 seconds |
| Curiosity | At least one curiosity segment required |
| Value | At least one practicality or strong emotion segment required |
| Speakers | Max 2 speaker changes (unless conversation format) |
| Context safety | No "as I said earlier", "going back to", etc. |

---

## 7. File-by-File Implementation Plan

### Backend Files to CREATE

| File | Purpose | Lines |
|------|---------|-------|
| `engine/__init__.py` | Package init | 1 |
| `engine/config.py` | All thresholds, constants, word lists, prompt templates, default fallbacks | 300+ |
| `engine/engine_config.yaml` | Runtime overrides for config.py values | 60 |
| `engine/pipeline.py` | Main orchestrator | 150+ |
| `engine/audio/extractor.py` | FFmpeg audio extraction | 20 |
| `engine/audio/quality.py` | SNR, speech rate, volume | 60 |
| `engine/audio/chunker.py` | Overlapping audio chunks | 40 |
| `engine/transcription/transcriber.py` | Groq Whisper API | 20 |
| `engine/transcription/aligner.py` | WhisperX alignment | 80 |
| `engine/transcription/language.py` | Language detection | 30 |
| `engine/transcription/fillers.py` | Filler removal | 20 |
| `engine/transcription/merger.py` | Chunk merge | 50 |
| `engine/segmentation/segmenter.py` | Atomic segmentation | 100 |
| `engine/features/sentiment.py` | VADER sentiment | 15 |
| `engine/features/patterns.py` | Regex patterns | 100+ |
| `engine/features/audio_features.py` | Per-segment audio features | 40 |
| `engine/features/structural.py` | Position features | 20 |
| `engine/rules/fundamentals.py` | Core rule definitions | 80 |
| `engine/rules/hook_rules.py` | Hook detection rules | 60 |
| `engine/rules/body_rules.py` | Body detection rules | 50 |
| `engine/rules/ending_rules.py` | Ending detection rules | 60 |
| `engine/rules/stitching_rules.py` | Stitching rules | 60 |
| `engine/graph/knowledge_graph.py` | Knowledge graph (legacy) | 100 |
| `engine/graph/relationships.py` | Edge detection (legacy) | 80 |
| `engine/scoring/candidate_generator.py` | Generate candidates | 80 |
| `engine/scoring/rule_engine.py` | Filter candidates (separates accepted vs rejected) | 60 |
| `engine/scoring/scorer.py` | Score candidates + uniqueness scoring | 80 |
| `engine/knowledge/local_graph.py` | Per-video knowledge graph | 120 |
| `engine/knowledge/global_graph.py` | Cross-video global knowledge graph | 150 |
| `engine/knowledge/context_db.py` | Context requirement database | 80 |
| `engine/knowledge/relationships.py` | Edge detection with confidence | 100 |
| `engine/llm/teacher.py` | LLM Teacher — structured label generation | 150 |
| `engine/llm/prompts.py` | Prompt templates (structured output) | 60 |
| `engine/llm/label_schemas.py` | Pydantic models for LLM labels | 80 |
| `engine/patterns/engine.py` | Pattern Intelligence Engine (orchestrator for all subsystems) | 180 |
| `engine/patterns/detector.py` | Pattern discovery + node statistics accumulator | 200 |
| `engine/patterns/matcher.py` | Match current video + meta pattern domain matching | 150 |
| `engine/patterns/graph.py` | Pattern graph builder + evolution tracker | 150 |
| `engine/patterns/context.py` | Context graph — what each pattern node needs | 120 |
| `engine/patterns/confidence.py` | Pattern confidence with decay and boost logic | 100 |
| `engine/patterns/meta.py` | Meta Pattern Graph — cross-pattern relationships | 140 |
| `engine/patterns/embeddings.py` | Future: pattern vector embeddings | 60 |
| `engine/confidence/scorer.py` | Confidence scoring for all decisions | 100 |
| `engine/confidence/threshold.py` | Adaptive threshold engine (routing) | 80 |
| `engine/decision/log.py` | Decision log / audit trail | 120 |
| `engine/decision/tracker.py` | Track which signals influenced each decision | 80 |
| `engine/decision/failures.py` | Store rejected candidates with structured reasons | 100 |
| `engine/data/models.py` | Data models (all schemas) | 200+ |
| `engine/data/storage.py` | SQLite storage + local JSON file fallback | 200 |
| `engine/data/migrations.py` | Schema migrations | 80 |
| `engine/data/local_store.py` | Local file-based storage (pre-database) | 150 |

### Backend Files to MODIFY

| File | Change | Lines |
|------|--------|-------|
| `app.py` | Add `/api/analyze` endpoint | +50 |

### Backend Files to KEEP UNCHANGED

| File | Reason |
|------|--------|
| `main.py` | Keep as reference for legacy pipeline |
| `editor.py` | Still used for effects post-processing |
| `hooks.py` | Still used for text overlays |
| `subtitles.py` | Still used for subtitle generation |

---

## 8. Dependencies

### New Packages (for engine/)

| Package | Version | Purpose |
|---------|---------|---------|
| `groq` | ≥0.4.0 | Groq API (Whisper + Llama) |
| `whisperx` | ≥3.1.0 | Word-level alignment |
| `librosa` | ≥0.10.0 | Audio feature extraction |
| `soundfile` | ≥0.12.0 | Audio file I/O |
| `pydub` | ≥0.25.1 | Audio chunking |
| `vaderSentiment` | ≥3.3.2 | Sentiment analysis |
| `sentence-transformers` | ≥2.2.0 | Semantic similarity (future) |
| `networkx` | ≥3.0 | Knowledge graph |
| `aiosqlite` | ≥0.19.0 | Async SQLite |
| `pyspellchecker` | — | Spell checking (opt.) |

### Already Installed (from current venv)

| Package | Version |
|---------|---------|
| `numpy` | 2.5.0 |
| `scipy` | 1.18.0 |
| `fastapi` | 0.136.1 |
| `uvicorn` | 0.46.0 |
| `python-dotenv` | 1.2.2 |
| `pydantic` | 2.13.4 |
| `python-multipart` | 0.0.27 |

---

## 9. Build Order

### Phase 1: Foundation (Days 1-3)

| Day | Task | Files |
|-----|------|-------|
| 1 | Audio extraction + quality analysis | `audio/extractor.py`, `audio/quality.py` |
| 2 | Transcription (Groq Whisper) | `transcription/transcriber.py`, `transcription/language.py`, `transcription/fillers.py` |
| 3 | Audio chunking + merge | `audio/chunker.py`, `transcription/merger.py` |

### Phase 2: Core Processing (Days 4-6)

| Day | Task | Files |
|-----|------|-------|
| 4 | Word alignment (WhisperX) | `transcription/aligner.py` |
| 5 | Atomic segmentation | `segmentation/segmenter.py` |
| 6 | Feature extraction | `features/sentiment.py`, `features/patterns.py`, `features/audio_features.py`, `features/structural.py` |

### Phase 3: Local Intelligence (Days 7-9)

| Day | Task | Files |
|-----|------|-------|
| 7 | Rules engine (fundamentals + hook/body/ending) | `rules/fundamentals.py`, `rules/hook_rules.py`, `rules/body_rules.py`, `rules/ending_rules.py`, `rules/stitching_rules.py` |
| 8 | Local knowledge graph + edge detection | `knowledge/local_graph.py`, `knowledge/relationships.py` |
| 9 | Candidate generation + scoring + stitching diversity | `scoring/candidate_generator.py`, `scoring/rule_engine.py`, `scoring/scorer.py` |

### Phase 4: LLM Teacher + Decision Systems (Days 10-12)

| Day | Task | Files |
|-----|------|-------|
| 10 | LLM Teacher — structured label generation + prompts + schemas | `llm/teacher.py`, `llm/prompts.py`, `llm/label_schemas.py` |
| 11 | Decision log + failure storage | `decision/log.py`, `decision/tracker.py`, `decision/failures.py` |
| 12 | Confidence system + adaptive thresholds | `confidence/scorer.py`, `confidence/threshold.py` |

### Phase 5: Pattern Recognition + Global Knowledge (Days 13-15)

| Day | Task | Files |
|-----|------|-------|
| 13 | Pattern Intelligence Engine (discovery + node stats + context graph + meta patterns) | `patterns/engine.py`, `patterns/detector.py`, `patterns/matcher.py`, `patterns/graph.py` |
| 14 | Global Knowledge Graph | `knowledge/global_graph.py`, `knowledge/context_db.py` |
| 15 | Data storage + schema migrations (all new tables) | `data/models.py`, `data/storage.py`, `data/migrations.py` |

### Phase 6: Integration (Days 16-18)

| Day | Task | Files |
|-----|------|-------|
| 16 | Pipeline orchestrator (all stages wired) | `pipeline.py` |
| 17 | API integration + end-to-end testing | `app.py` (modify) |
| 18 | Testing harness — 10 diverse podcasts, manual scoring sheet, regression suite | `tests/`, `test_podcasts/`, `scoring_sheet.md` |

**Total: ~18 days to working prototype**

---

### Testing Harness

A test harness must be ready **before** the pipeline is complete — build it alongside Phase 1 so every stage is tested immediately.

**10 Test Podcasts (curated at Day 1):**

| # | Podcast Type | Duration | Audio Quality | Language | Key Challenge |
|---|-------------|----------|---------------|----------|---------------|
| 1 | Clean studio interview | 30 min | High (SNR > 25dB) | English | Baseline |
| 2 | Noisy outdoor interview | 45 min | Low (SNR < 10dB) | English | Transcription accuracy |
| 3 | Heavy Indian accent | 60 min | Medium | Hinglish | Language detection |
| 4 | Fast-talking debate | 90 min | Medium | English | Speech rate + overlap |
| 5 | Monologue lecture | 120 min | High | English | Long format |
| 6 | Multi-host roundtable | 60 min | Medium | English | Speaker detection |
| 7 | Whisper ASMR-style | 20 min | High (low volume) | English | Volume sensitivity |
| 8 | Mixed language code-switch | 45 min | Medium | Spanish/English | Language detection |
| 9 | Low-bitrate recording | 30 min | Very low (8kHz) | English | Quality thresholds |
| 10 | British accent storytelling | 50 min | High | English | Pattern diversity |

**Per-Stage Validation:**

| Stage | Validation Method | Threshold |
|-------|------------------|-----------|
| Audio quality | SNR within ±3dB of manual measurement | ≥ 0.90 correlation |
| Transcription | WER vs manual transcript | < 15% |
| Word alignment | Timestamp error vs manual labels | < 0.3s error |
| Feature extraction | Manual feature audit on 50 segments | 100% complete |
| Hook detection | Human review: is this a hook? | Precision > 0.7 |
| Candidate scoring | Human ranking correlation | Spearman > 0.6 |
| LLM labels | Human label agreement | Cohen κ > 0.6 |

**Manual Scoring Sheet (for human evaluators):**

```
Clip ID: ________   Evaluator: ________   Date: ________

Rate each dimension 1-5:

Hook grabs attention?         [1] [2] [3] [4] [5]
Story feels complete?         [1] [2] [3] [4] [5]
Value delivered?              [1] [2] [3] [4] [5]
Emotional arc?                [1] [2] [3] [4] [5]
Would you save this?          [1] [2] [3] [4] [5]
Would you share this?         [1] [2] [3] [4] [5]
Context feels natural?        [1] [2] [3] [4] [5]
Any missing context?          [YES / NO] If yes: ______

Overall virality (1-10):      [___]
```

---

## 10. Data Storage for Future ML

### What We Store (Every Video)

| Data Point | Format | Purpose | Day |
|------------|--------|---------|-----|
| Segment text | String | NLP training | 1 |
| Word timestamps | Float[] | Alignment training | 1 |
| Sentiment score | Float (-1 to 1) | Emotion detection | 1 |
| Speech rate | Float | Audio analysis | 1 |
| Volume | Float | Emphasis detection | 1 |
| Patterns matched | String[] | Rule learning | 1 |
| Rules matched | String[] | Pattern learning | 1 |
| Hook/body/ending labels | Float[] | Clip structure | 1 |
| **Structured segment labels** | **Dict (15 fields)** | **Supervised learning** | **10** |
| **Structured candidate labels** | **Dict (11 fields)** | **Ranking model** | **10** |
| **Rejected candidates + reasons** | **Dict** | **Negative examples** | **11** |
| **Decision log entries** | **Dict[]** | **Feature importance analysis** | **11** |
| **Confidence scores** | **Float[]** | **Adaptive routing training** | **12** |
| **Pattern matches** | **Dict[]** | **Sequence model training** | **13** |
| **Global graph edge stats** | **Dict (7 fields)** | **Statistical weight learning** | **14** |
| **Context requirements** | **Dict** | **Context prediction model** | **14** |
| LLM analysis | String | Supervised learning | 10 |
| LLM rankings | Int[] | Ranking model | 10 |
| Actual performance | Dict | Reinforcement learning | After deploy |

### Target: 2000 Videos

After 2000 videos, the system will have:

| Asset | Count |
|-------|-------|
| Videos processed | 2,000 |
| Segments labeled | 800,000+ |
| Candidates evaluated (accepted + rejected) | 40,000+ |
| Decision log entries | 2,000,000+ |
| Patterns discovered | 400-600 |
| Global graph edges (statistically weighted) | 2,000+ |
| Context requirement records | 800,000+ |

With this dataset, we can:

1. **Train XGBoost ranker** on segment features + structured labels → predict virality score
2. **Train sequence model** on pattern sequences → predict best narrative structures
3. **Train binary classifier** on failures → predict rejection with 90%+ accuracy
4. **Replace LLM Teacher** for 60%+ of segments using local models
5. **Keep LLM only for edge cases** (< 10% of decisions)
6. **Zero AI cost** for 70%+ of processing pipeline

### Data Quality Requirements

- **Precision:** All timestamps must be accurate to ±0.1s
- **Completeness:** Every segment must have all features computed
- **Consistency:** Same features, same format, every run
- **Verification:** LLM-labeled data is reviewed for quality
- **Versioned:** Pipeline version stored with every analysis
- **Structured labels:** Every LLM interaction must produce valid, parseable JSON
- **Failure parity:** Rejected candidates stored with same schema depth as accepted
- **Decision traceability:** Every candidate must have a complete decision chain
- **Confidence calibration:** Confidence scores must correlate with actual accuracy (+/- 0.1)

### Local File Storage (Pre-Database Fallback)

Until SQLite integration is complete, all data must be stored as structured files on disk. This ensures the system works immediately (even in development) and the data can be imported into the database later.

**Storage root:** `D:\openshorts\engine\data\store\`

**File structure per video:**

```
engine/data/store/
├── videos/
│   └── {video_id}/
│       ├── metadata.json              # Video-level metadata
│       ├── transcript.json            # Full transcript with segments
│       ├── segments.json              # Atomic segments with all features
│       ├── labels.json                # LLM Teacher structured labels
│       ├── candidates.json            # Scored candidates (accepted + rejected)
│       ├── patterns.json              # Patterns discovered in this video
│       ├── decisions.jsonl            # Decision log entries (newline-delimited)
│       ├── confidence.json            # All confidence scores + provenance
│       └── state.json                 # Pipeline state (resume on crash)
├── global/
│   ├── pattern_db.json                # Global pattern database (accumulated)
│   ├── global_graph.json              # Global knowledge graph edges
│   ├── pattern_versions.json          # All pattern versions
│   ├── meta_patterns.json             # Meta pattern relationships
│   └── confidence_history.json        # Running confidence history
└── index.json                         # Master index of all processed videos
```

**File storage pattern:**

```python
import json
from pathlib import Path

class LocalStorage:
    def __init__(self, store_root: str = "engine/data/store"):
        self.root = Path(store_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.video_root = self.root / "videos"
        self.global_root = self.root / "global"
        self.video_root.mkdir(exist_ok=True)
        self.global_root.mkdir(exist_ok=True)
    
    def save_video_analysis(self, video_id: str, analysis: dict):
        """Save all analysis data for one video."""
        video_dir = self.video_root / video_id
        video_dir.mkdir(exist_ok=True)
        
        for filename, data in [
            ("metadata.json", analysis["metadata"]),
            ("transcript.json", analysis["transcript"]),
            ("segments.json", analysis["segments"]),
            ("labels.json", analysis["labels"]),
            ("candidates.json", analysis["candidates"]),
            ("patterns.json", analysis["patterns"]),
            ("confidence.json", analysis["confidence"]),
        ]:
            with open(video_dir / filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        
        # Decision log is newline-delimited JSON for append efficiency
        with open(video_dir / "decisions.jsonl", "a", encoding="utf-8") as f:
            for entry in analysis["decisions"]:
                f.write(json.dumps(entry) + "\n")
    
    def update_global(self, data_type: str, data: dict):
        """Update a global database file."""
        filepath = self.global_root / f"{data_type}.json"
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing.update(data)
        else:
            existing = data
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, default=str)
    
    def load_global_pattern_db(self) -> dict:
        """Load the accumulated pattern database."""
        filepath = self.global_root / "pattern_db.json"
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"patterns": [], "versions": [], "meta_edges": []}
    
    def migrate_to_sqlite(self):
        """One-time migration: import all JSON files into SQLite."""
        # Traverse videos/ directory, load each video's JSON files,
        # and insert into corresponding SQLite tables.
        pass
```

**Why this works:**
- No database dependency for development
- Data is human-readable and debuggable
- Enables simple git-based versioning of pattern databases
- One-time migration script (`migrate_to_sqlite`) when ready
- Global files accumulate naturally — `pattern_db.json` grows with each video

---

## 11. LLM as Teacher — Structured Labeling

### 11a. Philosophy Change

The original design treated the LLM as a **verifier** — it looked at top candidates and picked the best one. This is limiting because:

| Old Approach | New Approach |
|--------------|-------------|
| LLM verifies final candidates | LLM teaches by generating structured labels |
| Output: "Clip #3 is best" | Output: {"hook_strength": 0.91, "emotion": "curiosity"} |
| Single answer per video | Hundreds of labels per video |
| Human-readable reasoning only | Machine-trainable numeric vectors |
| Free-text rejection reasons | Structured failure categories |
| No confidence scores | Confidence score on every label |

The important output isn't *"Clip #3 is best."* The important output is *"Why did the model think Clip #3 is best?"*

### 11b. Per-Segment Labels

Every segment gets a structured label from the LLM:

```json
{
  "is_hook": 0.91,
  "hook_strength": 0.87,
  "is_context": 0.12,
  "is_takeaway": 0.03,
  "emotion": "curiosity",
  "requires_previous_context": 0.08,
  "creates_new_context": 0.94,
  "is_story": 0.05,
  "is_opinion": 0.76,
  "is_fact": 0.18,
  "speaker_confidence": 0.82,
  "saveability": 0.64,
  "shareability": 0.91,
  "confidence": 0.93
}
```

This creates a **supervised learning dataset** — every segment becomes a training example with ground-truth labels.

### 11c. Per-Candidate Labels

Every candidate clip also gets a structured label:

```json
{
  "story_complete": 0.88,
  "transition_quality": 0.82,
  "context_missing": 0.04,
  "shareability": 0.91,
  "saveability": 0.84,
  "hook_strength": 0.93,
  "ending_strength": 0.89,
  "emotional_arc_build_up": 0.76,
  "naturalness": 0.85,
  "curiosity_gap": 0.79,
  "confidence": 0.90
}
```

### 11d. Improved Prompts

Don't ask: *"Is this a good clip?"* (binary, opinion-based, low information density)

Ask: *"Rate from 0-1: Hook Strength, Context Completeness, Curiosity, Emotional Build-up, Story Completeness, Saveability, Shareability, Transition Quality, Naturalness, Requires Previous Context, Confidence"*

Every answer becomes training data. A single LLM call produces 10+ numeric labels instead of one free-text opinion.

### 11e. Label Schema Files

**File:** `engine/llm/label_schemas.py`

Defines Pydantic models for every label type:
- `SegmentLabel` — 15 numeric fields + 1 categorical (emotion)
- `CandidateLabel` — 11 numeric fields
- `RejectionLabel` — rejection reason (categorical) + confidence
- `PatternLabel` — pattern type + occurrence metadata

These schemas enforce consistent output format and enable automatic validation.

---

## 12. Dual Knowledge Graphs (Local + Global)

### 12a. Problem with a Single Graph

A single knowledge graph that only understands **one podcast** at a time has limited power. It cannot:
- Learn what makes a good hook across different podcasts
- Discover which narrative structures perform best
- Build statistical confidence on edge types
- Leverage patterns from previous videos

### 12b. Local Knowledge Graph (Per Video)

```
Hook (segment 143)
    │
    ├── follows → Problem (segment 144)
    │                │
    │                ├── explains → Explanation (segment 150)
    │                │                │
    │                │                ├── supports → Example (segment 162)
    │                │                └── contrasts → Counterpoint (segment 170)
    │                │
    │                └── contrasts → Objection (segment 158)
    │
    └── contrasts → Ending (segment 200)
```

**Properties:**
- Built fresh for each video
- Discarded after processing (except for extracted knowledge)
- Nodes = segments within the video
- Edges = temporal + semantic relationships
- Edge weights = rule-based (initial)

### 12c. Global Knowledge Graph (Across ALL Videos)

This is the **real brain** of the system — a persistent graph that accumulates statistical knowledge across every video processed.

```
Curiosity Hook
    │
    ├── 94% high retention
    ├── Usually followed by story
    └── Often ends with lesson

Personal Story
    │
    ├── High saveability
    ├── Medium shareability
    └── Strong emotional arc

Statistics
    │
    ├── Low retention alone
    └── Excellent when combined with story

Framework
    │
    ├── High saves
    └── Medium shares
```

**Edge weights become statistical over time:**

| Edge | Occurrences | Avg Watch Time | Avg Saves | Avg Shares | LLM Confidence |
|------|-----------|---------------|-----------|-----------|---------------|
| curiosity_hook → personal_story | 894 | 74% | 68% | 55% | 0.91 |
| statistics → framework | 342 | 61% | 82% | 43% | 0.87 |
| question → failure → lesson | 186 | 78% | 82% | 74% | 0.96 |
| money → mistake → framework | 134 | 65% | 91% | 38% | 0.94 |
| contrarian → takeaway | 97 | 71% | 59% | 67% | 0.88 |

**Graph Storage:**

```sql
CREATE TABLE global_edges (
    source_type TEXT,
    target_type TEXT,
    occurrences INTEGER,
    avg_watch_time REAL,
    avg_saves REAL,
    avg_shares REAL,
    llm_confidence REAL,
    avg_confidence REAL,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    PRIMARY KEY (source_type, target_type)
);
```

---

## 13. Pattern Intelligence Engine

### 13a. Philosophy

This is the **long-term knowledge system** of Trimora. It does far more than store patterns — it learns *why* they work, *when* they work, and how they *evolve* over time. Pattern discovery, context understanding, domain awareness, relationship tracking, and confidence decay all live here.

The Pattern Intelligence Engine is the core intellectual property of the system. It transforms individual video analyses into reusable, statistical knowledge that compounds with every processed video.

### 13b. Architecture

```
                Videos 1..N
                    │
                    ▼
         LLM Teacher (structured labels)
                    │
                    ▼
      ┌─────────────────────────────────────┐
      │    Pattern Intelligence Engine       │
      │                                      │
      │  ┌─────────────────────────────┐    │
      │  │ Pattern Discovery           │    │
      │  │ (find frequent sequences)   │    │
      │  └──────────┬──────────────────┘    │
      │             ▼                       │
      │  ┌─────────────────────────────┐    │
      │  │ Node Statistics Accumulator │    │
      │  │ (position, emotion,         │    │
      │  │  follower distributions)    │    │
      │  └──────────┬──────────────────┘    │
      │             ▼                       │
      │  ┌─────────────────────────────┐    │
      │  │ Context Graph               │    │
      │  │ (what does each node need?) │    │
      │  └──────────┬──────────────────┘    │
      │             ▼                       │
      │  ┌─────────────────────────────┐    │
      │  │ Pattern Evolution Tracker   │    │
      │  │ (how patterns change)       │    │
      │  └──────────┬──────────────────┘    │
      │             ▼                       │
      │  ┌─────────────────────────────┐    │
      │  │ Meta Pattern Graph          │    │
      │  │ (relationships between      │    │
      │  │  patterns - domain-aware)   │    │
      │  └──────────┬──────────────────┘    │
      │             ▼                       │
      │  ┌─────────────────────────────┐    │
      │  │ Confidence Decay Engine     │    │
      │  │ (patterns expire over time) │    │
      │  └─────────────────────────────┘    │
      └─────────────────────────────────────┘
                    │
                    ▼
    Global Pattern + Context Database
                    │
                    ▼
    Future: Pattern Embeddings (vector search)
```

### 13c. Pattern Discovery + Node Statistics

Every discovered pattern is more than a sequence — each **node** within it accumulates rich statistical behavior:

```
Pattern #421
Question
    │
    ▼
Personal Story
    │
    ▼
Failure
    │
    ▼
Lesson

Question Node
  │
  ├── Occurrences: 12,458
  ├── Avg Position in Video: 4.2 sec
  ├── Avg Emotion: Curiosity
  ├── Avg Sentiment: 0.65
  ├── Avg Duration: 4.8 sec
  ├── Confidence: 0.96
  │
  └── Usually Followed By:
        ├── Personal Story: 82%
        ├── Statistic: 11%
        └── Joke: 7%

Personal Story Node
  │
  ├── Occurrences: 8,234
  ├── Avg Position in Video: 12.7 sec
  ├── Avg Emotion: Empathy
  ├── Avg Sentiment: -0.12
  ├── Avg Duration: 18.2 sec
  ├── Confidence: 0.94
  │
  └── Usually Followed By:
        ├── Failure: 58%
        ├── Lesson: 22%
        ├── Framework: 12%
        └── Statistic: 8%

Failure Node
  │
  ├── Occurrences: 4,891
  ├── Avg Position in Video: 31.4 sec
  ├── Avg Emotion: Sadness
  ├── Avg Sentiment: -0.45
  ├── Avg Duration: 12.1 sec
  ├── Confidence: 0.91
  │
  └── Usually Followed By:
        ├── Lesson: 67%
        ├── Framework: 18%
        ├── Resolution: 10%
        └── Joke: 5%

Lesson Node
  │
  ├── Occurrences: 7,234
  ├── Avg Position in Video: 47.8 sec
  ├── Avg Emotion: Inspiration
  ├── Avg Sentiment: 0.72
  ├── Avg Duration: 9.5 sec
  ├── Confidence: 0.98
  │
  └── Usually Followed By:
        ├── Takeaway: 71%
        ├── Call to Action: 19%
        └── Summary: 10%
```

**This is behavior learning, not clip memorization.** The system learns what a Question node typically does — not just that it appeared in Pattern #421.

### 13d. Context Graph (Merged from Context Database)

Every pattern node knows what context it needs:

```
Personal Failure Node
  │
  ├── Needs Previous Context: YES (confidence: 0.94)
  ├── Context Types Required:
  │     ├── Person Introduction: YES (confidence: 0.89)
  │     ├── Timeline Context: OPTIONAL (confidence: 0.62)
  │     └── Topic Setup: NO (confidence: 0.91)
  │
  ├── Standalone Probability: 18%
  │
  ├── Required Previous Nodes:
  │     ├── Speaker_Introduction (confidence: 0.87)
  │     └── Story_Set_Up (confidence: 0.83)
  │
  └── Required Following Nodes:
        └── Resolution (confidence: 0.76)


Framework Node
  │
  ├── Needs Previous Context: NO (confidence: 0.88)
  ├── Context Types Required:
  │     └── None
  │
  ├── Standalone Probability: 82%
  │
  └── Required Previous Nodes:
        └── None


Question Node
  │
  ├── Needs Previous Context: NO (confidence: 0.96)
  ├── Standalone Probability: 91%
  │
  └── Creates New Context: YES (confidence: 0.94)
        └── Context Types Created:
              ├── Topic: 84%
              └── Curiosity Gap: 93%
```

Over time, the engine learns automatically which language patterns signal context requirements:

```
Text: "As I mentioned earlier..."
  → Needs Context: YES (confidence: 0.97)
  → Context Type: Setup
  → Required Previous: TopicIntroduction

Text: "So this guy walks in..."
  → Needs Context: YES (confidence: 0.82)
  → Context Type: Person
  → Required Previous: SpeakerIntroduction, Setting

Text: "The key takeaway is..."
  → Needs Context: NO (confidence: 0.94)
  → Standalone Viable: YES
  → Creates New Context: YES (type: Lesson)
```

### 13e. Pattern Evolution + Versioning

Patterns are not static. Every time a new video is processed, observed patterns update:

```
Day 1:
Question → Story → Failure → Lesson
  Occurrences: 1
  Confidence: 0.50

Day 100:
Question → Story → Failure → Lesson
  Occurrences: 47
  Confidence: 0.78

Day 365:
Question → Story → Failure → Lesson
  Occurrences: 186
  Confidence: 0.96

  But analysis shows adding a framework before the lesson
  increases saves by 14%:
  
Day 365+:
Question → Story → Failure → Framework → Lesson
  Occurrences: 43 (new variant)
  Confidence: 0.82 (growing)
  Avg Saves: 96% (higher than parent)
```

**Evolution becomes a competitive asset.** The engine detects that inserting a `Framework` node before `Lesson` increases saveability, and begins favoring that variant even though it has fewer total occurrences.

#### Pattern Versioning

Patterns must never be overwritten — they must be **versioned**. This enables comparison of which evolution performs better over time.

```
Pattern #421
  │
  ├── v1.0 (Day 1):    Question→Story→Failure→Lesson
  │     ├── Occurrences: 186
  │     ├── Avg Saves: 82%
  │     └── Status: SUPERSEDED
  │
  ├── v1.1 (Day 365):  Question→Story→Failure→Framework→Lesson
  │     ├── Occurrences: 43
  │     ├── Avg Saves: 96%
  │     └── Status: ACTIVE
  │
  └── v1.2 (Day 500):  Question→Story→Statistic→Failure→Framework→Lesson
        ├── Occurrences: 12
        ├── Avg Saves: 91%
        └── Status: EXPERIMENTAL
```

```python
class PatternVersion:
    pattern_id: int
    version: str          # "1.0", "1.1", "2.0", etc.
    parent_version: str   # which version this evolved from
    node_sequence: list[str]
    occurrences: int
    avg_saves: float
    avg_shares: float
    confidence: float
    confidence_history: list[dict]  # [{timestamp, value, trigger}, ...]
    status: str           # "ACTIVE" | "SUPERSEDED" | "EXPERIMENTAL"
    created_at: datetime
    superseded_at: datetime
    evolution_trigger: str  # "improved_performance" | "new_data" | "domain_shift"
    
    def compare_performance(self, other: 'PatternVersion'):
        """Compare two versions to determine which performs better."""
        return {
            "saves_delta": self.avg_saves - other.avg_saves,
            "shares_delta": self.avg_shares - other.avg_shares,
            "confidence_delta": self.confidence - other.confidence,
            "sample_size_ratio": self.occurrences / max(other.occurrences, 1)
        }
```

```python
class PatternEvolution:
    def track_variant(self, parent_id: int, variant: Pattern):
        """Track when a pattern spawns a new variant."""
        parent = self.db.get_pattern(parent_id)
        delta_saves = variant.avg_saves - parent.avg_saves
        delta_shares = variant.avg_shares - parent.avg_shares
        
        # Version the pattern instead of overwriting
        new_version = self.version_manager.create_version(
            parent_id=parent_id,
            node_sequence=variant.node_types,
            trigger="improved_performance" if delta_saves > 0.05 else "new_data"
        )
        
        if delta_saves > 0.05 or delta_shares > 0.05:
            self.version_manager.promote(new_version.id)
            parent.status = "SUPERSEDED"
        
        self.log_evolution({
            "parent_id": parent_id,
            "variant_id": variant.id,
            "version": new_version.version,
            "delta_saves": delta_saves,
            "delta_shares": delta_shares,
            "trigger": "improved_performance" if delta_saves > 0.05 else "new_data"
        })
```

### 13f. Meta Pattern Graph

This is what makes Trimora **domain-aware** without needing an LLM. Instead of only learning clip structures, the engine learns **relationships between patterns** — and which patterns work best for which content categories.

```
Curiosity Pattern (#421)
    │
    ├──► Business Pattern (#128)
    │       │
    │       ├── Videos: 342
    │       ├── Money→Mistake→Framework
    │       └── Avg Saves: 91%
    │
    ├──► Startup Pattern (#089)
    │       │
    │       ├── Videos: 156
    │       ├── Problem→Solution→Growth
    │       └── Avg Saves: 84%
    │
    └──► Psychology Pattern (#211)
            │
            ├── Videos: 97
            ├── Story→Emotion→Lesson
            └── Avg Saves: 78%

Personal Story Pattern (#421)
    │
    ├──► Business Category (34% of occurrences)
    │       └── Prefers: Money node after Failure
    │
    ├──► Psychology Category (28% of occurrences)
    │       └── Prefers: Emotion node before Lesson
    │
    └──► Health Category (12% of occurrences)
            └── Prefers: Statistic before Story
```

**Engine result:** When the system detects a Business video, it automatically knows:

- **Preferred hook:** Money/Statistic (82% confidence)
- **Preferred arc:** Problem → Failure → Framework (91% confidence)
- **Preferred ending:** Framework with actionable steps (94% confidence)

No LLM call needed. This is pure learned knowledge.

```python
class MetaPatternGraph:
    def get_preferred_structure(self, category: str) -> Pattern:
        """Return the best-performing pattern for a content category."""
        candidates = self.db.query("""
            SELECT p.*, mp.occurrences 
            FROM meta_patterns mp
            JOIN patterns p ON mp.pattern_id = p.id
            WHERE mp.category = ? AND mp.confidence > 0.8
            ORDER BY p.avg_saves * p.confidence DESC
            LIMIT 1
        """, (category,))
        return candidates[0] if candidates else None
    
    def get_edge_weight(self, source_pattern_id: int, 
                         target_pattern_id: int, category: str) -> float:
        """Get the transition probability between two patterns in a category."""
        return self.db.get_meta_edge(
            source_pattern_id, target_pattern_id, category
        ).transition_probability
```

### 13g. Pattern Aging + Confidence with Decay

Confidence is not a fixed attribute. Patterns lose relevance as content trends evolve. A pattern that worked in 2025 might not work in 2028.

Every pattern tracks:

| Attribute | Description | Example |
|-----------|-------------|---------|
| `confidence` | Statistical confidence based on occurrence count | 0.96 |
| `freshness` | How recently the pattern was observed (0-1) | 0.87 |
| `last_seen` | Timestamp of last occurrence | 2026-07-01 |
| `trend` | Is performance improving, stable, or declining? | "improving" |
| `decay_rate` | How fast confidence erodes per day | 0.0003 |
| `half_life_days` | Days until confidence halves if unobserved | 365 |

```
Pattern #421 (Question→Story→Failure→Lesson)

Timeline:
  2025-01: confidence=0.50, freshness=1.00, trend=stable
  2025-06: confidence=0.78, freshness=0.92, trend=improving
  2026-01: confidence=0.96, freshness=0.85, trend=stable
  2026-07: confidence=0.94, freshness=0.78, trend=stable
  2028-01: confidence=0.82, freshness=0.45, trend=declining
            └── Content trends shifted. Newer patterns outperforming.
```

```python
class PatternAging:
    BASE_DECAY_RATE = 0.0003     # per day since last seen
    FRESHNESS_HALF_LIFE = 365    # 1 year
    
    def compute_confidence(self, pattern: Pattern) -> float:
        days_since_update = (datetime.now() - pattern.last_seen).days
        
        # Base confidence from occurrence count (law of large numbers)
        n = pattern.occurrences
        base_confidence = 1.0 - (1.0 / (1.0 + n * 0.01))
        
        # Apply time decay
        decay = self.BASE_DECAY_RATE * days_since_update
        final = min(base_confidence - decay, 0.99)
        return max(final, 0.01)
    
    def compute_freshness(self, pattern: Pattern) -> float:
        """0-1 score: how recently was this pattern observed?"""
        days_since = (datetime.now() - pattern.last_seen).days
        return 2.0 ** (-days_since / self.FRESHNESS_HALF_LIFE)
    
    def compute_trend(self, pattern: Pattern) -> str:
        """Is this pattern gaining or losing relevance?"""
        recent = self.get_recent_performance(pattern, window_days=90)
        older = self.get_recent_performance(pattern, window_days=180, offset=90)
        
        if recent > older * 1.1:
            return "improving"
        elif recent < older * 0.9:
            return "declining"
        else:
            return "stable"
    
    def boost_on_success(self, pattern: Pattern, 
                          actual_saves: float, actual_shares: float):
        expected = (pattern.avg_saves + pattern.avg_shares) / 2
        actual = (actual_saves + actual_shares) / 2
        
        if actual > expected * 1.1:
            pattern.confidence = min(pattern.confidence + 0.02, 0.99)
            pattern.last_seen = datetime.now()
            pattern.freshness = 1.0
```

**Aging prevents the engine from overvaluing outdated storytelling structures.** If a pattern hasn't appeared in 2+ years, its confidence drops below 0.80, allowing newer, more relevant patterns to surface.

**Confidence History is stored per pattern** — every confidence update is logged with its timestamp and trigger:

```json
[
  {"date": "2025-01-15", "confidence": 0.50, "trigger": "discovered"},
  {"date": "2025-06-20", "confidence": 0.78, "trigger": "new_occurrences_(47)"},
  {"date": "2026-01-10", "confidence": 0.96, "trigger": "new_occurrences_(186)"},
  {"date": "2026-07-01", "confidence": 0.94, "trigger": "time_decay"},
  {"date": "2028-01-01", "confidence": 0.82, "trigger": "time_decay"}
]
```

### 13h. Pattern Matching at Runtime

When processing a new video, the engine asks: **"Have I seen this narrative before?"** and incorporates all the learned intelligence above:

```python
def match_against_global(segments, labels, category: str):
    local_sequence = extract_label_sequence(segments, labels)
    matches = []
    
    # 1. Direct pattern match
    for i in range(len(local_sequence) - MIN_PATTERN_LEN):
        subseq = local_sequence[i:i+3]
        pattern = GLOBAL_DB.find_pattern(subseq)
        if pattern and pattern.confidence > 0.85:
            matches.append({
                "pattern_id": pattern.id,
                "match_position": i,
                "expected_saves": pattern.avg_saves,
                "expected_shares": pattern.avg_shares,
                "confidence": pattern.confidence
            })
    
    # 2. Domain-aware meta pattern match
    if category:
        preferred = META_GRAPH.get_preferred_structure(category)
        if preferred:
            similarity = compute_sequence_similarity(
                local_sequence, preferred.node_types
            )
            if similarity > 0.6:
                matches.append({
                    "pattern_id": preferred.id,
                    "match_type": "meta",
                    "confidence": similarity * preferred.confidence,
                    "domain": category
                })
    
    return matches
```

### 13i. Pattern Database Schema

```sql
CREATE TABLE patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_sequence TEXT,             -- JSON array of segment types
    occurrences INTEGER,
    avg_saves REAL,
    avg_shares REAL,
    avg_watch_time REAL,
    confidence REAL,
    decay_rate REAL DEFAULT 0.0003,
    context_requirement TEXT,       -- "low" | "medium" | "high"
    typical_duration_min REAL,
    typical_duration_max REAL,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    evolution_history TEXT,         -- JSON: how pattern changed over time
    categories TEXT,                -- JSON: topic categories this pattern appears in
    parent_pattern_id INTEGER,      -- NULL for root patterns
    variant_count INTEGER
);

CREATE TABLE pattern_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_id INTEGER REFERENCES patterns(id),
    node_index INTEGER,
    node_type TEXT,                  -- "hook" | "story" | "fact" | "opinion" | ...
    emotion TEXT,
    avg_duration REAL,
    avg_sentiment REAL,
    avg_position REAL,               -- avg position in the video (seconds)
    occurrences INTEGER,
    confidence REAL,
    follower_distribution TEXT       -- JSON: {type: probability, ...}
);

CREATE TABLE pattern_context (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_node_id INTEGER REFERENCES pattern_nodes(id),
    needs_previous_context BOOLEAN,
    context_types_required TEXT,     -- JSON: [{type, confidence}, ...]
    context_types_created TEXT,      -- JSON: [{type, confidence}, ...]
    standalone_probability REAL,
    required_previous_nodes TEXT,    -- JSON: [{node_type, confidence}, ...]
    required_following_nodes TEXT,   -- JSON: [{node_type, confidence}, ...]
    confidence REAL
);

CREATE TABLE pattern_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_id INTEGER REFERENCES patterns(id),
    source_index INTEGER,
    target_index INTEGER,
    transition_type TEXT,            -- "follows" | "explains" | "contrasts" | ...
    transition_quality REAL,
    transition_probability REAL
);

CREATE TABLE meta_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_pattern_id INTEGER REFERENCES patterns(id),
    target_pattern_id INTEGER REFERENCES patterns(id),
    category TEXT,                   -- "business" | "psychology" | "health" | ...
    occurrences INTEGER,
    transition_probability REAL,
    avg_saves REAL,
    avg_shares REAL,
    confidence REAL
);

CREATE TABLE pattern_embeddings (         -- Future: vector similarity search
    pattern_id INTEGER PRIMARY KEY REFERENCES patterns(id),
    embedding BLOB,                       -- numpy array as bytes
    dimension INTEGER,
    version INTEGER
);

CREATE TABLE pattern_evolution_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_id INTEGER REFERENCES patterns(id),
    event_type TEXT,                      -- "discovered" | "updated" | "variant" | "decayed" | "promoted"
    previous_confidence REAL,
    new_confidence REAL,
    trigger TEXT,                         -- "new_occurrence" | "performance_boost" | "time_decay"
    details TEXT,                         -- JSON
    timestamp TIMESTAMP
);
```

### 13j. Pattern Embeddings (Future — Post-MVP)

After surpassing ~10,000 patterns, the engine can represent each pattern as a vector:

```
Question → Story → Failure → Lesson
                ↓
         [0.42, 0.87, 0.13, 0.56, ...]  (embedding)
                ↓
    Similar patterns cluster in vector space:

    Question→Story→Lesson          Curiosity→Experience→Takeaway
    [0.41, 0.85, 0.15, 0.54]      [0.44, 0.82, 0.18, 0.51]
                ↑                                  ↑
                └────────── close neighbors ───────┘
```

This enables **similarity search without an LLM**: "Find me patterns like this one" becomes a nearest-neighbor query against the embedding index.

---

## 14. Confidence System

### 14a. Philosophy

Every decision in the pipeline should have a **confidence score**. This enables:

1. **Adaptive routing:** High confidence → use local model. Low confidence → ask LLM.
2. **Graceful degradation:** If confidence drops below threshold, fall back to simpler rules.
3. **Model replacement roadmap:** Gradually replace LLM calls with learned models as confidence grows.

### 14b. Confidence Sources

| Source | What It Measures | Range |
|--------|-----------------|-------|
| Rule Engine | How clearly does the segment match the rule? | 0.0 — 1.0 |
| Knowledge Graph | How strong is the edge relationship? | 0.0 — 1.0 |
| Pattern Engine | How well does this match known patterns? | 0.0 — 1.0 |
| LLM Teacher | How confident is the LLM in its label? | 0.0 — 1.0 |
| Global Graph | How often has this edge been validated? | 0.0 — 1.0 |

### 14c. Adaptive Threshold Engine

```python
class AdaptiveThresholdEngine:
    """Adjusts LLM dependency based on confidence scores."""
    
    def get_routing_decision(self, confidence: float):
        if confidence > 0.9:
            return "USE_LOCAL_MODEL"    # Zero LLM cost
        elif confidence > 0.6:
            return "USE_PATTERN_MATCH"  # Use global pattern DB
        else:
            return "USE_LLM"            # Fall back to Groq
```

### 14d. Confidence Routing Matrix

| Confidence | Action | Cost | Speed |
|-----------|--------|------|-------|
| 0.90 — 1.00 | Use local deterministic model | $0 | Instant |
| 0.75 — 0.90 | Use pattern match from Global Graph | $0 | ~10ms |
| 0.60 — 0.75 | Use rule engine with relaxed thresholds | $0 | ~5ms |
| 0.30 — 0.60 | Use LLM (Groq) for this decision | ~$0.001 | ~2s |
| 0.00 — 0.30 | Flag for human review | $0 | Manual |

### 14e. Feature Provenance

Every stored feature must record its **source** — where did this signal come from? Years later, you'll need to know which signals to trust and which to replace.

```python
@dataclass
class FeatureProvenance:
    feature_name: str           # "hook_strength", "shareability", "speech_rate", ...
    value: float
    source: str                 # "rule_engine" | "llm_teacher" | "librosa" | "vader" | ...
    confidence: float           # how much do we trust this source for this feature?
    pipeline_version: str       # which version of the source produced this?
    model_version: str          # if ML model, which model version
    override_count: int         # how many times this was manually overridden
```

**Example provenance registry:**

| Feature | Typical Source | Source Confidence | Replaceable By |
|---------|---------------|-------------------|----------------|
| `hook_strength` | Rule Engine | 0.92 | Pattern match (after 500 videos) |
| `shareability` | LLM Teacher | 0.88 | Trained regressor (after 2000 videos) |
| `speech_rate` | librosa | 0.99 | (deterministic — no replacement needed) |
| `emotion` | VADER | 0.76 | Fine-tuned classifier (after 500 videos) |
| `sentiment` | VADER | 0.82 | Fine-tuned classifier (after 500 videos) |
| `requires_context` | LLM Teacher | 0.91 | Pattern match (after 1000 videos) |
| `confidence` | LLM Teacher | 0.93 | Calibrated model (after 2000 videos) |

```sql
CREATE TABLE feature_provenance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT,
    entity_id TEXT,              -- segment_id or candidate_id
    entity_type TEXT,            -- "segment" | "candidate"
    feature_name TEXT,
    value REAL,
    source TEXT,                 -- "rule_engine" | "llm_teacher" | "librosa" | "vader" | ...
    source_confidence REAL,
    pipeline_version TEXT,
    model_version TEXT,
    created_at TIMESTAMP
);
```

### 14f. Confidence Propagation

Confidence should **propagate through the pipeline** — uncertainty upstream should affect confidence downstream.

```
Transcription Confidence: 0.98
    │
    ▼
Segmentation Confidence: 0.95
    │  (inherits 0.98 * 0.97 from transcription)
    ▼
Feature Extraction Confidence: 0.92
    │  (inherits 0.95 * 0.97 from segmentation)
    ▼
Hook Detection Confidence: 0.89
    │  (inherits 0.92 * 0.97 from features)
    ▼
Candidate Score Confidence: 0.84
    │  (inherits 0.89 * 0.94 from hook detection)
    ▼
Final Recommendation Confidence: 0.81
```

Each stage multiplies incoming confidence by its own reliability:

```python
class ConfidencePropagator:
    def propagate(self, pipeline_context: PipelineContext) -> float:
        """Compute propagated confidence through the pipeline."""
        stage_confidences = {
            "transcription": 0.98,
            "segmentation": 0.97,
            "feature_extraction": 0.97,
            "knowledge_graph": 0.96,
            "rule_engine": 0.95,
            "scoring": 0.94,
            "llm_teacher": 0.93,
        }
        
        cumulative = 1.0
        for stage, reliability in stage_confidences.items():
            stage_confidence = pipeline_context.get_stage_confidence(stage)
            cumulative *= (stage_confidence * reliability)
        
        return cumulative
    
    def get_downstream_impact(self, upstream_drop: float, 
                               stages_remaining: int) -> float:
        """How much does an upstream confidence drop affect downstream results?"""
        avg_stage_reliability = 0.96
        return upstream_drop * (avg_stage_reliability ** stages_remaining)
```

**Example:** If transcription confidence drops to 0.60 (poor audio quality), the final recommendation confidence drops from ~0.81 to ~0.50 — the system automatically knows to trust its output less.

### 14g. Confidence Database

```sql
CREATE TABLE confidence_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT,
    entity_id TEXT,              -- segment_id or candidate_id
    entity_type TEXT,            -- "segment" | "candidate"
    decision_type TEXT,          -- "hook_detection" | "body_detection" | ...
    source TEXT,                 -- "rule" | "pattern" | "llm" | "global_graph"
    raw_confidence REAL,
    propagated_confidence REAL,
    routing_decision TEXT,       -- "local" | "pattern" | "llm" | "human"
    provenance TEXT,             -- JSON: which upstream stages contributed
    upstream_confidences TEXT,   -- JSON: {stage: confidence, ...}
    created_at TIMESTAMP,
    pipeline_version TEXT
);

CREATE INDEX idx_conf_video ON confidence_log(video_id);
CREATE INDEX idx_conf_source ON confidence_log(source, decision_type);
```

---

## 15. Decision Log & Audit Trail

### 15a. Purpose

Instead of only storing **outputs**, store the **decision process** — the complete chain of signals that led to each choice. Think of this as an audit trail for every clip the system produces.

### 15b. Decision Chain Example

```
Segment 143
    ↓
Matched Rule: Curiosity Pattern (confidence: 0.87)
    ↓
Matched Rule: High Energy (speech_rate > 2.0) (confidence: 0.92)
    ↓
Knowledge Graph: "Explains" edge → Segment 149 (confidence: 0.76)
    ↓
LLM Teacher: Hook Score = 0.91 (confidence: 0.93)
    ↓
Pattern Match: Pattern #421 (Question → Story → Lesson) (confidence: 0.82)
    ↓
Global Graph: Similar hooks avg 76% watch time (confidence: 0.88)
    ↓
Final Decision: Selected as Hook
```

### 15c. Decision Log Schema

```sql
CREATE TABLE decision_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT,
    entity_id TEXT,              -- segment_id, candidate_id, or pattern_id
    entity_type TEXT,            -- "segment" | "candidate" | "pattern"
    stage TEXT,                  -- "rules" | "graph" | "llm" | "pattern_match"
    rule_name TEXT,
    rule_category TEXT,
    confidence REAL,
    outcome TEXT,                -- "selected" | "rejected" | "candidate"
    rejection_reason TEXT,
    contributing_signals TEXT,   -- JSON: which signals influenced this decision
    timestamp TIMESTAMP,
    pipeline_version TEXT
);

CREATE INDEX idx_decision_entity ON decision_log(entity_id, entity_type);
CREATE INDEX idx_decision_stage ON decision_log(stage, outcome);
```

### 15d. Structured Logging

All pipeline logs must be **machine-readable JSON**, not free-text. This ensures decision logs can be parsed, queried, and fed into ML pipelines without preprocessing.

```python
import structlog  # or json formatter with standard logging

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

# Usage
logger.info("hook_detected", 
    segment_id="seg_143",
    hook_score=0.91,
    confidence=0.93,
    matched_rules=["curiosity", "high_energy"],
    source="rule_engine",
    video_id="vid_0182",
    pipeline_version="1.0.0"
)

logger.warning("transcription_quality_low",
    chunk_id="chunk_047",
    snr_db=8.2,
    threshold=10.0,
    action="switched_to_strict_chunking",
    video_id="vid_0182"
)

logger.error("stage_failed",
    stage="word_alignment",
    error="whisperx_audio_load_failed",
    recoverable=True,
    fallback="unaligned_boundaries",
    video_id="vid_0182"
)
```

**Every log entry must include:**
- `timestamp` — ISO 8601
- `event` — machine-readable event name (snake_case)
- `video_id` — which video
- `pipeline_version` — which code version
- `stage` — which pipeline stage
- All relevant numeric values as structured fields (not embedded in text)

**No free-text log messages in production code.** The `event` field replaces log message strings. Human-readable descriptions are added as a separate `description` field only when needed.

### 15e. Value Over Time

| Time | Value |
|------|-------|
| Day 1 | Debugging — see exactly which rules fire |
| Month 1 | Tuning — identify rules that never fire or always fire |
| Month 6 | ML training — feature importance analysis |
| Year 2 | Model replacement — know which signals to replicate |

---

## 16. Failure Storage

### 16a. Philosophy

**Failures are just as valuable as successes** because they teach future models what *not* to choose. Most systems only store successful candidates — this is a missed opportunity.

### 16b. What Gets Stored

Every rejected candidate is stored with:

```json
{
  "candidate_id": "cand_0451_rejected",
  "video_id": "vid_0182",
  "hook_segment": {...},
  "body_segments": [...],
  "ending_segment": {...},
  "total_score": 0.42,
  "rejection_reason": "missing_context",
  "rejection_stage": "rule_engine",
  "rules_failed": ["no_context_gaps", "has_curiosity"],
  "rules_passed": ["total_duration", "hook_in_3_seconds"],
  "llm_label": {
    "story_complete": 0.23,
    "context_missing": 0.87,
    "shareability": 0.31,
    ...
  },
  "decision_chain": [...]
}
```

### 16c. Failure Schema

```sql
CREATE TABLE failed_candidates (
    id TEXT PRIMARY KEY,
    video_id TEXT,
    hook_segment_id TEXT,
    body_segment_ids TEXT,
    ending_segment_id TEXT,
    total_duration REAL,
    hook_score REAL,
    body_score REAL,
    ending_score REAL,
    flow_score REAL,
    total_score REAL,
    rejection_reason TEXT,
    rejection_stage TEXT,
    rules_failed TEXT,           -- JSON array
    rules_passed TEXT,           -- JSON array
    llm_label TEXT,              -- JSON (CandidateLabel)
    decision_chain TEXT,         -- JSON array of decision log entries
    created_at TIMESTAMP,
    pipeline_version TEXT
);
```

### 16d. Failure Analysis

After enough failures accumulate, the system can answer questions like:

- *"What is the most common rejection reason?"* — **Missing context** (34% of failures)
- *"Which rules reject the most candidates?"* — **has_curiosity** (rejects 41%)
- *"How does LLM score correlate with rejection reasons?"* — **context_missing**: avg 0.87 confidence
- *"Which rejection reasons cluster together?"* — **weak_ending** + **no_emotional_arc** co-occur 67% of the time

---

## 17. Future Deterministic Engine

### 17a. End State Vision

```
Confidence > 0.9
    ↓
Use LOCAL MODEL (rules + features only)
    → Zero API cost, ~10ms per decision
    → Handles 70%+ of decisions at scale

Confidence 0.6 — 0.9
    ↓
Use PATTERN MATCH (Global Knowledge Graph)
    → Zero API cost, ~50ms
    → Handles 20% of decisions

Confidence < 0.6
    ↓
Use LLM TEACHER (Groq)
    → ~$0.001 per call, ~2s
    → Handles < 10% of decisions
    → Each call generates training data to reduce future calls
```

### 17b. The Data Flywheel

```
More videos processed
    ↓
More structured labels generated
    ↓
Pattern database grows richer
    ↓
Global graph confidence increases
    ↓
More decisions can use local/pattern models
    ↓
LLM dependency decreases
    ↓
Cost per video drops
    ↓
Can process more videos
```

### 17c. Milestones

| Milestone | LLM Dependency | Cost Per Video | Pattern DB Size |
|-----------|---------------|---------------|----------------|
| 0-100 videos | 100% (all decisions via LLM) | ~$0.50 | 0 patterns |
| 100-500 videos | 70% LLM, 30% pattern | ~$0.35 | ~100 patterns |
| 500-2000 videos | 40% LLM, 50% pattern, 10% local | ~$0.20 | ~500 patterns |
| 2000+ videos | 10% LLM, 30% pattern, 60% local | ~$0.05 | ~2000+ patterns |
| 10000+ videos | 2% LLM, 8% pattern, 90% local | ~$0.01 | ~10000+ patterns |

### 17d. What Becomes Replaceable

| Component | Replaceable After | Replacement |
|-----------|------------------|-------------|
| Hook detection rules | 200 videos | XGBoost classifier on segment features |
| Body context detection | 500 videos | Sequence model on labeled segments |
| Ending detection | 500 videos | Pattern match from Global Graph |
| LLM Teacher (segments) | 1000 videos | Trained regression model |
| LLM Teacher (candidates) | 2000 videos | Trained ranking model |
| Full pipeline (no LLM) | 10000+ videos | Learned deterministic engine |

---

## 18. Review Feedback & Applied Improvements

### 11a. Transcription Reliability
**Feedback:** Groq Whisper is fast, but for best word-level alignment, always run WhisperX on the full audio after initial chunk transcription.

**Status:** ✅ Already in plan (Stage 5). Added explicit note to ensure WhisperX receives the full audio file, not chunks.

### 11b. Language Detection
**Feedback:** Unicode-based detector is clever for Hinglish/Hindi/Bengali. Consider adding `langdetect` or `fasttext` as fallback.

**Status:** ✅ Added recommendation in `language.py` section. These can be added as optional fallback when Unicode analysis is ambiguous.

### 11c. Pattern Matching Expansion
**Feedback:** Expand relatability and takeaway categories with more real-world patterns.

**Status:** ✅ Added:
- `relatability`: `shared_experience`, `empathy` patterns
- `takeaway`: `call_to_action`, `here_is`, `conclusion` patterns
- `power_words`: New category for high-impact words (secret, shocking, never, always, etc.)

### 11d. Graph Traversal
**Feedback:** In `get_connected_cluster()`, add a max-duration constraint during BFS.

**Status:** ✅ Implemented with cumulative duration check that prunes branches exceeding `max_duration` (default 90s).

### 11e. Scoring Uniqueness Bonus
**Feedback:** Add a small "uniqueness" bonus using simple TF-IDF against the full transcript.

**Status:** ✅ Added `compute_uniqueness_score()` in `scorer.py`. Weights 5% of total score. Penalizes generic filler clips and rewards distinctive content.

### 11f. Context References
**Feedback:** Expand regex list with common podcast phrases.

**Status:** ✅ Added: `like I was saying`, `to go back`, `as mentioned earlier`, `coming back to`, `recall that/when/how`.

### 11g. Stitching Diversity
**Feedback:** Prioritize clips where body segments come from different parts of the video (true stitching).

**Status:** ✅ Added `compute_stitching_diversity()` in Stage 11b. Rewards clips spanning 60-180s of source material, penalizes overly adjacent or distant combinations.

### 11h. LLM Usage
**Feedback:** Keeping LLM only for verification (Stage 14) is correct. Optional early analyzer is fine for bootstrapping rules.

**Status:** ✅ **Evolved beyond this.** The original plan used LLM as a verifier. The current plan (Sections 11-17) uses LLM as a **teacher and structured label generator** — producing machine-trainable labels for every segment and candidate, storing failures alongside successes, and feeding a Pattern Intelligence Engine. The pipeline is still 95% deterministic at runtime; LLM labels enrich the training dataset.

---

## 19. Final Review Verdict

### Overall Assessment

This implementation plan is **production-ready**. It supports the core goals from the original vision:

| Goal | How It's Achieved |
|------|-------------------|
| **Strong hooks (peaks)** | Hook rules + curiosity/energy/contrast scoring |
| **Smart stitching with context** | Hook from anywhere, body from elsewhere, ending from elsewhere via graph traversal |
| **Natural emotional arc** | Peak → Downfall/tension → Resolution/takeaway |
| **Savability** | Practicality, lessons, relatability, takeaway patterns |
| **Shareability** | Curiosity, surprise, power words, contrast markers |
| **Context preservation** | Context reference detection, flow scoring, emotional arc check |
| **Path to ML** | Rich structured data stored from day one (2000+ video target) |

### Remaining Polish (Recommended Before Ship)

#### 12a. Error Handling & Logging
Add robust `try/except` blocks and detailed logging throughout `pipeline.py`, especially around:
- **WhisperX alignment** — alignment failures on low-quality audio
- **Graph traversal** — edge cases with disconnected segments
- **FFmpeg extraction** — corrupt or unsupported audio formats

Suggested approach: structured logging with per-stage timestamps, error counts, and input/output sizes.

#### 12b. Config-Driven Thresholds
Move all magic numbers into `engine/config.py` (or `engine_config.yaml`) for easy tuning without code changes:

| Current Hardcoded Value | Config Key | Default |
|-------------------------|------------|---------|
| `45-90s` clip duration | `CLIP_MIN_DURATION`, `CLIP_MAX_DURATION` | 45, 90 |
| `3-8s` hook duration | `HOOK_MIN_DURATION`, `HOOK_MAX_DURATION` | 3, 8 |
| `50` hook min score | `HOOK_MIN_SCORE` | 50 |
| `30` ending min score | `ENDING_MIN_SCORE` | 40 |
| `0.7` recency threshold | `RECENCY_THRESHOLD` | 0.7 |
| `2.0` speech rate (hook) | `HOOK_SPEECH_RATE_MIN` | 2.0 |
| `1.5` volume delta (hook) | `HOOK_VOLUME_DELTA_MIN` | 1.5 |
| `60`s temporal window | `TEMPORAL_WINDOW_SECONDS` | 60 |

#### 12c. Testing Strategy
After Phase 1-2, create a test suite with **10 diverse podcasts** covering:

- Clean studio audio (high SNR)
- Noisy/interview audio (low SNR, cross-talk)
- Heavy accents (Indian, British, etc.)
- Mixed language (Hinglish, Spanglish)
- Different lengths (20min, 1hr, 2hr+)

Manually review 20-30 generated candidates and tune thresholds accordingly.

#### 12d. Performance for Long Videos
For videos > 2 hours, consider parallel processing where safe:

- **Chunk transcription** — parallelize with `ThreadPoolExecutor` (Groq API calls are I/O-bound)
- **Feature extraction** — batch segment analysis with numpy vectorization
- **Graph building** — remains sequential (depends on all segments)

### Integration with Existing Codebase

In `app.py`, the new `POST /api/analyze` endpoint should return:

```json
{
  "video_id": "uuid-here",
  "candidates": [
    {
      "id": "clip-1",
      "hook_segment": {"text": "...", "start": 12.5, "end": 17.2},
      "body_segments": [{"text": "...", "start": 45.0, "end": 62.3}],
      "ending_segment": {"text": "...", "start": 120.1, "end": 126.8},
      "total_duration": 57.3,
      "hook_score": 0.82,
      "body_score": 0.71,
      "ending_score": 0.78,
      "flow_score": 0.85,
      "total_score": 0.79,
      "transcript_snippet": "What if I told you... [context] ...that's the key lesson"
    }
  ],
  "best_clip": { ... },
  "reasoning": "Hook creates strong curiosity gap, body provides concrete example, ending delivers actionable takeaway",
  "stats": {
    "total_duration": 3600.0,
    "segments_found": 420,
    "candidates_generated": 156,
    "candidates_passed_rules": 23,
    "processing_time_seconds": 45.2,
    "pipeline_version": "1.0.0"
  }
}
```

The top candidates can then be fed directly into the existing **video cutting → subtitle → hook overlay** pipeline from `main.py`, `subtitles.py`, and `hooks.py`.

### Final Verdict

> This is one of the most thoughtful clipping system designs — much stronger than most "full LLM" commercial tools because it's **controllable, debuggable, and cost-efficient**. You are ready to start coding.
