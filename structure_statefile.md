# Application Structure Statefile

_Last updated: October 10, 2025_

## Root Directory

- `data/`
  - `english-corpora.org/`
    - `coca/`
      - `text_acad_isi/`
      - `text_blog_wih/`
      - `text_fic_jjw/`
      - `text_mag_jgr/`
      - `text_news_nne/`
      - `text_spok_yuv/`
      - `text_tvm_nwh/`
      - `text_web_ywv/`
    - `sample/`
      - `coca-samples-db/`
      - `coca-samples-text/`
  - `loc.gov/`
    - `scotus_pdfs/`
  - `scdb.la.psu.edu/`
- `dist/`
- `examples/`
  - `ai/`
    - `reports/`
  - `coca/`
  - `embedding/`
  - `multi-lingual/`
  - `scotus/`
  - `topic_modeling/`
- `getout_of_text_3/`
  - `__pycache__/`
- `getout_of_text_3.egg-info/`
- `img/`

---

**Notes:**
- This structure omits the `app`, `iac`, and `scripts` directories for clarity.
- Subdirectories are shown to two levels where relevant.
- This file should be kept up to date and referenced by all AI agents for context alignment and change tracking.
