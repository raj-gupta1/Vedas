---
license: other
license_name: public-domain
license_link: LICENSE
language:
- en
- sa
- hi
tags:
- Vedas
- Hindu
- History
- India
- RigVeda
- SamaVeda
- YajurVeda
- AtharvaVeda
size_categories:
- 10K<n<100K
data_prepared_from:
- Ralph T.H. Griffith's Rig Veda, Sama Veda, White Yajur Veda, and Atharva Veda English translations
- Arthur Berriedale Keith's Black Yajur Veda English translation
Chunk_Overlap:
- 150 characters
Chunk_Size:
- Full Hymn / Anuvaka / Verse granularity
---

# 📖 Four Vedas Corpus (Rig Veda, Sama Veda, Yajur Veda, Atharva Veda)

**Languages:** English (en)
**Dataset size:** ~10K < n < ~100K records (chunks and verses combined)

---

## Dataset Summary

This corpus is a comprehensive, highly structured dataset comprising the four primary pillars of ancient Indian literature: the **Rig Veda**, **Sama Veda**, **Yajur Veda**, and **Atharva Veda**. It integrates multiple authoritative translations, offering Sanskrit source verses alongside English and Hindi interpretations.

The dataset is formatted for continuous pre-training, fine-tuning, and Retrieval-Augmented Generation (RAG) applications in Indology, computational linguistics, historical analysis, and ancient language model development.

---

## 🏛️ Corpus Composition & Authorship

The dataset brings together works from highly respected translators and scholars:

1. **Rig Veda**: 
   - **Translator**: **Ralph T.H. Griffith** (1896).
   - **Language**: English (en).
   - **Structure**: 10 Mandalas (Books), organized by Suktas (Hymns) and verses.
   
2. **Sama Veda**:
   - **Translators**: 
     - **Ralph T.H. Griffith** (1895) - English translation.
     - **PDF Manuscripts** - High-resolution digital renderings.
   - **Languages**: English (en) | Sanskrit (sa) | Hindi (hi).
   - **Structure**: Part I (Mula/Decades) & Part II (Hymns).

3. **Yajur Veda**:
   - **Translators**:
     - **Black Yajur Veda** (Taittiriya Samhita): **Arthur Berriedale Keith** (1914) - English.
     - **White Yajur Veda** (Vajasaneya Samhita): **Ralph T.H. Griffith** (1899) - English.
   - **Languages**: English (en).
   - **Structure**: Kandas & Prapathakas (Black) | 40 Books & Verses (White).

4. **Atharva Veda**:
   - **Translators**:
     - **Ralph T.H. Griffith** (1895) - English.
   - **Languages**: English (en).
   - **Structure**: 20 Books, Suktas, and Verses.

---

## ⚙️ Chunking & Preprocessing Strategies

Two specialized pipelines are used to prepare the Vedic data for LLM ingestion:

### 1. Document Page Chunking (PDF pipeline)
Used for the bilingual translation documents (e.g., Sharma's Atharva Veda and PDF versions of Sama Veda) to ensure balanced context sizes.

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Chunk Overlap** | 150 characters | Ensures smooth transitions and preserves semantic boundaries |
| **Chunk Size** | 4 chunks per page | Optimizes sequence length for embedding and context retrieval |
| **Strategy** | Page-level overlapping splits | Preserves adjacent translation pairs (Sanskrit-Hindi) |

### 2. Semantic Structural Scraping (Web pipeline)
Used for the web-scraped books to preserve original ritualistic, musical, and narrative divisions:

*   **Rig Veda & Atharva Veda**: Chunked at the **Hymn (Sukta)** level. This encapsulates full structural themes as single database rows.
*   **Sama Veda**: Chunked at the **Decade/Hymn** level, aligning with Vedic musical metres.
*   **Black Yajur Veda**: Chunked at the **Anuvaka** level, maintaining ritual instruction blocks.
*   **White Yajur Veda**: Chunked at the individual **Verse** level with high-resolution metadata tags.

---

## 📁 Data Structure

Dataset records follow a unified JSON schema containing the text payload alongside hierarchical metadata:

### Scraped Structural Record Example (JSONL)
```json
{
  "text": "BLACK YAJUR VEDA (TAITTIRIYA SAMHITA)\nKANDA I\nPRAPATHAKA I\nANUVAKA i. 1. 1.\nThe New and Full Moon Sacrifices\n\na For food thee, for strength thee!\nb Ye are winds, ye are approachers.\nc Let the god Savitr impel you to the most excellent offering...",
  "source": "sacred-texts.com",
  "collection": "Yajur Veda",
  "sub_collection": "Black Yajur Veda",
  "translator": "Arthur Berriedale Keith",
  "kanda": 1,
  "prapathaka": 1,
  "anuvaka": "i. 1. 1.",
  "title": "The New and Full Moon Sacrifices"
}
```

---

## 🚀 Purpose & Applications

*   **Ancient Text Pre-Training**: Ideal for injecting Vedic domain knowledge into deep learning models (e.g., VedaGPT).
*   **Bilingual RAG (Retrieval-Augmented Generation)**: High-quality translations allow cross-lingual semantic searches between English, Hindi, and Sanskrit.
*   **Linguistic & Digital Humanities**: Facilitates statistical studies of ancient Sanskrit syntax, meter, and translation structures.
