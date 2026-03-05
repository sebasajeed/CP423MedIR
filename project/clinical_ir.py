import argparse
import json
import math
import os
import tempfile
from typing import Any

import soundfile as sf
import torch
from dotenv import load_dotenv
from groq import Groq
from pyannote.audio import Pipeline
from sentence_transformers import SentenceTransformer
from supabase import create_client


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


class ClinicalIRSystem:
    def __init__(self) -> None:
        print("--- Initializing Clinical IR System (Python) ---")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        groq_api_key = _required_env("GROQ_API_KEY")
        supabase_url = _required_env("SUPABASE_URL")
        supabase_key = _required_env("SUPABASE_KEY")
        hf_auth_token = _required_env("HF_AUTH_TOKEN")

        self.groq_client = Groq(api_key=groq_api_key)
        self.supabase = create_client(supabase_url, supabase_key)
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        self.hf_auth_token = hf_auth_token
        self.diarization_pipeline = None

    def _ensure_diarization_pipeline(self) -> None:
        if self.diarization_pipeline is None:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1", token=self.hf_auth_token
            ).to(self.device)

    def process_audio_file(self, audio_path: str, role_mapping: dict[str, str]) -> None:
        self._ensure_diarization_pipeline()
        print(f"Step 1: Reading {audio_path}...")
        data, samplerate = sf.read(audio_path)
        waveform = torch.tensor(data).float()

        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.T

        audio_payload = {"waveform": waveform, "sample_rate": samplerate}

        print("Step 2: Identifying speakers (Diarization)...")
        diar_output = self.diarization_pipeline(audio_payload)
        diar_segments = []
        for turn, speaker in diar_output.exclusive_speaker_diarization:
            diar_segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})

        print("Step 3: Transcribing with Groq Whisper...")
        with open(audio_path, "rb") as file:
            transcription = self.groq_client.audio.transcriptions.create(
                file=(audio_path, file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )

        print("Step 4: Speaker-aware indexing to Supabase...")
        for w_seg in transcription.segments:
            midpoint = (w_seg["start"] + w_seg["end"]) / 2
            current_speaker = "UNKNOWN"
            for d_seg in diar_segments:
                if d_seg["start"] <= midpoint <= d_seg["end"]:
                    current_speaker = d_seg["speaker"]
                    break

            role = role_mapping.get(current_speaker, "OTHER")
            text = w_seg["text"].strip()
            self.index_segment(
                content=text,
                speaker_role=role,
                metadata={"start": w_seg["start"], "end": w_seg["end"]},
            )

    def index_segment(
        self, content: str, speaker_role: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        clean_text = content.strip()
        if not clean_text:
            raise ValueError("Cannot index an empty transcript segment.")
        embedding = self.embed_model.encode(clean_text).tolist()
        payload = {
            "content": clean_text,
            "speaker_role": speaker_role,
            "embedding": embedding,
            "metadata": metadata or {},
        }
        response = self.supabase.table("clinical_segments").insert(payload).execute()
        records = response.data or []
        return records[0] if records else payload

    def transcribe_audio_bytes(self, audio_bytes: bytes, filename: str = "chunk.wav") -> list[dict]:
        if not audio_bytes:
            raise ValueError("Audio bytes are empty.")

        suffix = os.path.splitext(filename)[1] or ".wav"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name

            with open(tmp_path, "rb") as file:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=(filename, file.read()),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                )

            segments = []
            for seg in transcription.segments:
                segments.append(
                    {
                        "start": float(seg["start"]),
                        "end": float(seg["end"]),
                        "text": seg["text"].strip(),
                    }
                )
            return segments
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _fetch_segments(self, role_filter: str = "ALL") -> list[dict]:
        query = self.supabase.table("clinical_segments").select(
            "id, speaker_role, content, embedding, metadata"
        )
        if role_filter != "ALL":
            query = query.eq("speaker_role", role_filter)
        response = query.execute()
        return response.data or []

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return -1.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return -1.0
        return dot / (norm_a * norm_b)

    def search_segments(
        self, query_text: str, top_k: int = 5, role_filter: str = "ALL"
    ) -> list[dict]:
        segments = self._fetch_segments(role_filter=role_filter)
        if not segments:
            return []

        q_embedding = self.embed_model.encode(query_text).tolist()
        scored = []
        for seg in segments:
            score = self._cosine_similarity(q_embedding, seg.get("embedding", []))
            scored.append(
                {
                    "id": seg.get("id"),
                    "speaker_role": seg.get("speaker_role"),
                    "content": seg.get("content"),
                    "metadata": seg.get("metadata", {}),
                    "score": score,
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def answer_question(
        self, question: str, top_k: int = 5, role_filter: str = "ALL"
    ) -> tuple[str, list[dict]]:
        retrieved = self.search_segments(
            query_text=question, top_k=top_k, role_filter=role_filter
        )
        if not retrieved:
            return "No indexed segments found for this query.", []

        context = "\n".join(
            [
                f"- [{seg['speaker_role']}] (score={seg['score']:.3f}) {seg['content']}"
                for seg in retrieved
            ]
        )
        prompt = f"""
Answer the question using ONLY the retrieved transcript segments.
If the answer is not present, say "Not enough evidence in retrieved segments."

Question:
{question}

Retrieved Segments:
{context}
"""
        completion = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content, retrieved

    def get_full_transcript(self) -> str:
        print("\n--- Fetching full speaker-separated transcript ---")
        response = (
            self.supabase.table("clinical_segments")
            .select("speaker_role, content, metadata")
            .order("metadata->start", desc=False)
            .execute()
        )

        transcript_text = ""
        records = response.data or []
        for record in records:
            line = f"[{record['speaker_role']}]: {record['content']}"
            print(line)
            transcript_text += line + "\n"
        return transcript_text

    def generate_clinical_summary(self, transcript: str) -> str:
        print("\n--- Generating LLM summary ---")
        prompt = f"""
Summarize the following clinical interview.
Focus on patient concerns and clinician observations.

TRANSCRIPT:
{transcript}

SUMMARY FORMAT:
1. Patient Reported Symptoms:
2. Clinician Observations/Questions:
3. Follow-up Plan:
"""
        completion = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content

    def evaluate_retrieval(self, qrels_path: str, top_k: int = 5) -> dict:
        with open(qrels_path, "r", encoding="utf-8") as f:
            qrels = json.load(f)

        if not isinstance(qrels, list):
            raise ValueError("Evaluation file must be a JSON list of query specs.")

        overall = []
        by_role: dict[str, list[dict]] = {}

        for item in qrels:
            query_text = item.get("query", "").strip()
            relevant_contents = set(item.get("relevant_contents", []))
            role_filter = item.get("role_filter", "ALL")

            if not query_text or not relevant_contents:
                continue

            retrieved = self.search_segments(
                query_text=query_text, top_k=top_k, role_filter=role_filter
            )
            retrieved_contents = [r["content"] for r in retrieved]
            hits = sum(1 for c in retrieved_contents if c in relevant_contents)

            precision = hits / top_k if top_k > 0 else 0.0
            recall = hits / len(relevant_contents) if relevant_contents else 0.0

            result = {
                "query": query_text,
                "role_filter": role_filter,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "hits": hits,
            }
            overall.append(result)
            by_role.setdefault(role_filter, []).append(result)

        def avg(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        overall_summary = {
            "num_queries": len(overall),
            "avg_precision_at_k": avg([r["precision_at_k"] for r in overall]),
            "avg_recall_at_k": avg([r["recall_at_k"] for r in overall]),
        }
        role_summary = {}
        for role, results in by_role.items():
            role_summary[role] = {
                "num_queries": len(results),
                "avg_precision_at_k": avg([r["precision_at_k"] for r in results]),
                "avg_recall_at_k": avg([r["recall_at_k"] for r in results]),
            }

        return {"overall": overall_summary, "by_role": role_summary, "details": overall}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clinical IR audio pipeline")
    parser.add_argument("--audio-file", default="audio.wav", help="Path to interview audio file")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run diarization + transcription + indexing before summary generation",
    )
    parser.add_argument("--search-query", help="Run semantic retrieval for a query")
    parser.add_argument("--qa-query", help="Answer question grounded in retrieved segments")
    parser.add_argument("--evaluate-file", help="Path to qrels JSON for Precision@K/Recall@K")
    parser.add_argument("--top-k", type=int, default=5, help="Top K for retrieval/evaluation")
    parser.add_argument(
        "--role-filter",
        choices=["ALL", "PATIENT", "CLINICIAN", "OTHER"],
        default="ALL",
        help="Restrict retrieval by speaker role",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    roles = {"SPEAKER_00": "CLINICIAN", "SPEAKER_01": "PATIENT"}

    bot = ClinicalIRSystem()

    if args.ingest:
        bot.process_audio_file(args.audio_file, roles)

    if args.search_query:
        print(f"\n--- SEARCH RESULTS (top {args.top_k}, role={args.role_filter}) ---")
        results = bot.search_segments(
            query_text=args.search_query, top_k=args.top_k, role_filter=args.role_filter
        )
        if not results:
            print("No segments found.")
            return
        for idx, r in enumerate(results, start=1):
            print(f"{idx}. [{r['speaker_role']}] score={r['score']:.3f} | {r['content']}")
        return

    if args.qa_query:
        print(f"\n--- QA (top {args.top_k}, role={args.role_filter}) ---")
        answer, supporting = bot.answer_question(
            question=args.qa_query, top_k=args.top_k, role_filter=args.role_filter
        )
        print("\nAnswer:\n")
        print(answer)
        print("\nSupporting Segments:\n")
        for idx, seg in enumerate(supporting, start=1):
            print(f"{idx}. [{seg['speaker_role']}] score={seg['score']:.3f} | {seg['content']}")
        return

    if args.evaluate_file:
        print(f"\n--- EVALUATION @K={args.top_k} ---")
        summary = bot.evaluate_retrieval(args.evaluate_file, top_k=args.top_k)
        print(json.dumps(summary, indent=2))
        return

    full_transcript = bot.get_full_transcript()
    if not full_transcript.strip():
        print("No segments found in 'clinical_segments'. Run with --ingest first.")
        return
    summary = bot.generate_clinical_summary(full_transcript)
    print("\n--- CLINICAL SUMMARY ---\n")
    print(summary)


if __name__ == "__main__":
    main()
