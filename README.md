# Session Text Analysis

This project analyzes text editing sessions to classify sentences based on authorship (user-written, AI-generated, or both). It processes `.jsonl` files and outputs per-session `.csv` files with sentence-level annotations.

---

## Project Structure

Analysis/
├── files/ # (Optional) raw data or source files
├── output_csvs/ # Output CSVs per input session
├── main.py # Main script (optional or experimental)
├── process_all_sessions.py # Batch processing script
├── output_csvs.zip # Compressed CSV output (optional)
├── sentences_by_author.csv # (Optional) combined or summary output
├── venv/ # Python virtual environment

## How to Run

1. **Place your `.jsonl` session files** in a directory (e.g., `./1/`)
2. **Run the batch processor:**

```bash
python process_all_sessions.py
python main.py
```

3. **Output files will be saved into the output_csvs/ directory — one .csv per session file.**
