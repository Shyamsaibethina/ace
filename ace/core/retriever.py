from playbook_utils import parse_playbook_line, format_playbook_line
from sentence_transformers import SentenceTransformer
import numpy as np


class Retriever:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large", top_k: int = 5):
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k

    def retrieve(self, playbook: str, question: str, context: str = "", top_k: int = None) -> str:
        top_k = top_k if top_k is not None else self.top_k
        bullets_with_sections = self._extract_bullets_with_sections(playbook)
        if not bullets_with_sections:
            return ""

        top_k = min(top_k, len(bullets_with_sections))

        passage_texts = [
            "passage: " + b["content"] for b in bullets_with_sections
        ]
        passage_embeddings = self.model.encode(passage_texts, normalize_embeddings=True)

        query_text = "query: " + question + " " + context
        query_embedding = self.model.encode(query_text, normalize_embeddings=True)

        similarities = np.dot(passage_embeddings, query_embedding)
        top_k_indices = set(np.argsort(similarities)[-top_k:])

        section_order = list(dict.fromkeys(
            b["section"] for b in bullets_with_sections
        ))

        section_bullets: dict[str, list[str]] = {s: [] for s in section_order}
        for i, b in enumerate(bullets_with_sections):
            if i in top_k_indices:
                line = format_playbook_line(b["id"], b["helpful"], b["harmful"], b["content"])
                section_bullets[b["section"]].append(line)

        lines = []
        for section in section_order:
            if section_bullets[section]:
                lines.append(section)
                lines.extend(section_bullets[section])
                lines.append("")

        print("Used bullets:")
        for i, b in enumerate(bullets_with_sections):
            if i in top_k_indices:
                print(f"  {b['id']}: {b['content']}")

        print("Out of total bullets:")
        print(f"  {len(bullets_with_sections)}")

        return "\n".join(lines).rstrip()
    
    # To make sure we go by sections and we keep the section order
    def _extract_bullets_with_sections(self, playbook: str) -> list:
        results = []
        current_section = ""
        for line in playbook.strip().split("\n"):
            if line.strip().startswith("##"):
                current_section = line.strip()
            else:
                parsed = parse_playbook_line(line)
                if parsed:
                    parsed["section"] = current_section
                    results.append(parsed)
        return results
