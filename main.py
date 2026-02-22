"""Simple AI Resume Screener CLI.

Run this script to rank PDF resumes in a folder against a job description.

Example:
	python main.py --resumes-dir resumes --job "Senior Python developer with ML experience"

The script expects an OpenAI API key in the environment variable `OPENAI_API_KEY`.
You can create a `.env` file with `OPENAI_API_KEY=sk-...` and this project uses `python-dotenv`.
"""

import argparse
import os
from typing import List

from resume_screener import extractor, embeddings, similarity


def main(resumes_dir: str, job_description: str) -> None:
	if not os.path.isdir(resumes_dir):
		raise SystemExit(f"Resumes directory not found: {resumes_dir}")

	print("Extracting text from PDFs...")
	texts_map = extractor.extract_texts_from_dir(resumes_dir)
	if not texts_map:
		print("No PDF resumes found in the directory.")
		return

	filenames: List[str] = []
	resume_texts: List[str] = []
	for fname, text in texts_map.items():
		filenames.append(fname)
		resume_texts.append(text or "")

	print("Generating embeddings (this requires OPENAI_API_KEY)...")
	try:
		resume_embs = embeddings.embed_texts(resume_texts)
		job_emb = embeddings.get_embedding(job_description)
	except Exception as e:
		raise SystemExit(f"Embedding error: {e}")

	# Compute similarity scores and store results as a list of dicts
	results = []
	for i, fname in enumerate(filenames):
		emb = resume_embs[i]
		score = similarity.cosine_similarity(job_emb, emb)
		results.append({"filename": fname, "text": resume_texts[i], "score": float(score)})

	# Sort by descending similarity
	results.sort(key=lambda r: r["score"], reverse=True)

	print("\nRanked candidates:")
	for rank, item in enumerate(results, start=1):
		print(f"{rank:2d}. {item['filename']}: {item['score']:.4f}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="AI Resume Screener")
	parser.add_argument("--resumes-dir", default="resumes", help="Folder containing PDF resumes")
	parser.add_argument("--job", help="Job description text. If omitted, will prompt.")
	args = parser.parse_args()

	job = args.job
	if not job:
		job = input("Enter job description: \n")

	main(args.resumes_dir, job)


