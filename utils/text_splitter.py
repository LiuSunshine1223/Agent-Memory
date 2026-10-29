"""按句子边界对基础知识文档进行分块。"""

import re


class SentenceBoundaryChunker:
    def __init__(self, max_chunk_size, overlap_sentences):
        self.max_chunk_size = int(max_chunk_size)
        self.overlap_sentences = int(overlap_sentences)
        if self.max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        if self.overlap_sentences < 0:
            raise ValueError("overlap_sentences cannot be negative")

    def split_text(self, text):
        if not text:
            return []

        pattern = r"[^。！？.!?\n]+(?:[。！？.!?\n]+|$)"
        sentences = re.findall(pattern, text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        chunks = []
        current_sentences = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if sentence_len > self.max_chunk_size:
                if current_sentences:
                    chunks.append("".join(current_sentences))
                    current_sentences = []
                    current_length = 0
                for start in range(0, sentence_len, self.max_chunk_size):
                    chunks.append(
                        sentence[start : start + self.max_chunk_size]
                    )
                continue

            if (
                current_sentences
                and current_length + sentence_len > self.max_chunk_size
            ):
                chunks.append("".join(current_sentences))
                current_sentences = current_sentences[
                    -self.overlap_sentences :
                ] if self.overlap_sentences else []
                while (
                    current_sentences
                    and sum(len(item) for item in current_sentences)
                    + sentence_len
                    > self.max_chunk_size
                ):
                    current_sentences.pop(0)
                current_length = sum(
                    len(item) for item in current_sentences
                )

            current_sentences.append(sentence)
            current_length += sentence_len

        if current_sentences:
            chunks.append("".join(current_sentences))

        return chunks
