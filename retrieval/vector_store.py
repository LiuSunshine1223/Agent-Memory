"""FAISS storage with aligned text and discrete-memory metadata."""

from __future__ import annotations

import copy
import os
import pickle
from threading import RLock

import faiss
import numpy as np


class VectorStore:
    """Persist vectors, texts and metadata in one single-process store."""

    SNAPSHOT_VERSION = 1

    def __init__(self, storage_path, dimension):
        self.storage_path = storage_path
        self.dimension = int(dimension)
        self.snapshot_file = os.path.join(storage_path, "store.snapshot")
        self._lock = RLock()

        os.makedirs(storage_path, exist_ok=True)
        if os.path.exists(self.snapshot_file):
            self._load_snapshot()
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.texts = []
            self.metadatas = []

    def _load_snapshot(self):
        with open(self.snapshot_file, "rb") as source:
            snapshot = pickle.load(source)
        if not isinstance(snapshot, dict):
            raise ValueError("invalid vector store snapshot")
        if snapshot.get("version") != self.SNAPSHOT_VERSION:
            raise ValueError(
                f"unsupported snapshot version: {snapshot.get('version')}"
            )
        stored_dimension = int(snapshot.get("dimension", -1))
        if stored_dimension != self.dimension:
            raise ValueError(
                f"stored index dimension {stored_dimension} "
                f"!= configured dimension {self.dimension}"
            )

        serialized_index = np.asarray(
            snapshot.get("index"),
            dtype="uint8",
        )
        self.index = faiss.deserialize_index(serialized_index)
        self.texts = list(snapshot.get("texts", []))
        self.metadatas = list(snapshot.get("metadatas", []))
        self._validate_alignment()

    def add_texts(self, texts, embeddings, metadatas=None):
        if not texts:
            return []

        texts = list(texts)
        embeddings_np = np.asarray(embeddings, dtype="float32")
        if embeddings_np.ndim == 1:
            embeddings_np = embeddings_np.reshape(1, -1)
        if embeddings_np.ndim != 2:
            raise ValueError("embeddings must be a 1D or 2D array")
        if len(texts) != len(embeddings_np):
            raise ValueError("texts and embeddings must have the same length")
        if embeddings_np.shape[1] != self.dimension:
            raise ValueError(
                f"embedding dimension {embeddings_np.shape[1]} != {self.dimension}"
            )

        if metadatas is None:
            metadatas = [{} for _ in texts]
        metadatas = [dict(item or {}) for item in metadatas]
        if len(metadatas) != len(texts):
            raise ValueError("texts and metadatas must have the same length")

        with self._lock:
            start = len(self.texts)
            previous = (
                faiss.deserialize_index(faiss.serialize_index(self.index)),
                list(self.texts),
                copy.deepcopy(self.metadatas),
            )
            try:
                self.index.add(embeddings_np)
                self.texts.extend(texts)
                for offset, metadata in enumerate(metadatas):
                    metadata.setdefault("status", "active")
                    metadata.setdefault("kind", "knowledge")
                    metadata.setdefault("slot", f"knowledge.{start + offset}")
                    self.metadatas.append(metadata)
                self.save()
            except Exception:
                self.index, self.texts, self.metadatas = previous
                raise
            return list(range(start, start + len(texts)))

    def replace_all(self, texts, embeddings, metadatas=None):
        """Replace the complete store and persist it as one snapshot.

        Document updates and privacy-driven hard deletion need physical index
        rebuilds because ``IndexFlatL2`` has no stable external record IDs.
        The previous in-memory state is restored when persistence fails.
        """

        texts = list(texts)
        embeddings_np = np.asarray(embeddings, dtype="float32")
        if not texts:
            embeddings_np = np.empty((0, self.dimension), dtype="float32")
        elif embeddings_np.ndim == 1:
            embeddings_np = embeddings_np.reshape(1, -1)
        if embeddings_np.ndim != 2:
            raise ValueError("embeddings must be a 1D or 2D array")
        if len(texts) != len(embeddings_np):
            raise ValueError("texts and embeddings must have the same length")
        if embeddings_np.shape[1] != self.dimension:
            raise ValueError(
                f"embedding dimension {embeddings_np.shape[1]} != {self.dimension}"
            )

        if metadatas is None:
            metadatas = [{} for _ in texts]
        metadatas = [dict(item or {}) for item in metadatas]
        if len(metadatas) != len(texts):
            raise ValueError("texts and metadatas must have the same length")
        for record_id, metadata in enumerate(metadatas):
            metadata.setdefault("status", "active")
            metadata.setdefault("kind", "knowledge")
            metadata.setdefault("slot", f"knowledge.{record_id}")

        replacement = faiss.IndexFlatL2(self.dimension)
        if len(embeddings_np):
            replacement.add(embeddings_np)

        with self._lock:
            previous = (self.index, self.texts, self.metadatas)
            self.index = replacement
            self.texts = texts
            self.metadatas = metadatas
            try:
                self.save()
            except Exception:
                self.index, self.texts, self.metadatas = previous
                raise

    def purge_records(self, filters):
        """Physically remove matching records and rebuild the flat index."""

        if not filters:
            raise ValueError("purge_records requires at least one filter")
        with self._lock:
            removed_ids = [
                record_id
                for record_id, metadata in enumerate(self.metadatas)
                if self._matches(metadata, filters)
            ]
            if not removed_ids:
                return 0

            removed = set(removed_ids)
            retained_ids = [
                record_id
                for record_id in range(len(self.texts))
                if record_id not in removed
            ]
            texts = [self.texts[record_id] for record_id in retained_ids]
            metadatas = [
                copy.deepcopy(self.metadatas[record_id])
                for record_id in retained_ids
            ]
            embeddings = (
                np.asarray(
                    [self.index.reconstruct(record_id) for record_id in retained_ids],
                    dtype="float32",
                )
                if retained_ids
                else np.empty((0, self.dimension), dtype="float32")
            )
            self.replace_all(texts, embeddings, metadatas)
            return len(removed_ids)

    def search(self, query_vector, recall_k, max_distance=None):
        return [
            item["text"]
            for item in self.search_records(
                query_vector,
                recall_k=recall_k,
                max_distance=max_distance,
            )
        ]

    def search_records(
        self,
        query_vector,
        recall_k,
        max_distance=None,
        filters=None,
    ):
        if recall_k <= 0:
            return []

        query_np = np.asarray(query_vector, dtype="float32")
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)
        if query_np.ndim != 2 or query_np.shape[0] != 1:
            raise ValueError("query_vector must contain exactly one vector")
        if query_np.shape[1] != self.dimension:
            raise ValueError(
                f"query dimension {query_np.shape[1]} != {self.dimension}"
            )

        with self._lock:
            if self.index.ntotal == 0:
                return []
            search_k = self.index.ntotal if filters else min(
                int(recall_k), self.index.ntotal
            )
            distances, indices = self.index.search(query_np, search_k)
            results = []
            for position, record_id in enumerate(indices[0]):
                if record_id == -1 or record_id >= len(self.texts):
                    continue
                distance = float(distances[0][position])
                if max_distance is not None and distance > max_distance:
                    continue
                metadata = self.metadatas[record_id]
                if not self._matches(metadata, filters):
                    continue
                results.append(
                    {
                        "record_id": int(record_id),
                        "text": self.texts[record_id],
                        "metadata": copy.deepcopy(metadata),
                        "distance": distance,
                    }
                )
                if len(results) >= recall_k:
                    break
            return results

    def list_records(self, filters=None):
        with self._lock:
            return [
                {
                    "record_id": record_id,
                    "text": text,
                    "metadata": copy.deepcopy(metadata),
                }
                for record_id, (text, metadata) in enumerate(
                    zip(self.texts, self.metadatas)
                )
                if self._matches(metadata, filters)
            ]

    def update_metadata(self, record_id, updates, persist=True):
        with self._lock:
            if record_id < 0 or record_id >= len(self.metadatas):
                raise IndexError(f"invalid record_id: {record_id}")
            previous = copy.deepcopy(self.metadatas[record_id])
            try:
                self.metadatas[record_id].update(dict(updates))
                if persist:
                    self.save()
            except Exception:
                self.metadatas[record_id] = previous
                raise

    def update_metadatas(self, updates_by_id):
        with self._lock:
            previous = copy.deepcopy(self.metadatas)
            for record_id, updates in updates_by_id.items():
                if record_id < 0 or record_id >= len(self.metadatas):
                    raise IndexError(f"invalid record_id: {record_id}")
            try:
                for record_id, updates in updates_by_id.items():
                    self.metadatas[record_id].update(dict(updates))
                if updates_by_id:
                    self.save()
            except Exception:
                self.metadatas = previous
                raise

    def save(self):
        with self._lock:
            snapshot = {
                "version": self.SNAPSHOT_VERSION,
                "dimension": self.dimension,
                "index": faiss.serialize_index(self.index),
                "texts": self.texts,
                "metadatas": self.metadatas,
            }
            self._atomic_pickle(self.snapshot_file, snapshot)

    @property
    def total_count(self):
        return self.index.ntotal

    def _validate_alignment(self):
        if self.index.d != self.dimension:
            raise ValueError(
                f"stored index dimension {self.index.d} "
                f"!= configured dimension {self.dimension}"
            )
        if self.index.ntotal != len(self.texts):
            raise ValueError(
                "snapshot vector count and text count are inconsistent"
            )
        if len(self.metadatas) != len(self.texts):
            raise ValueError(
                "snapshot metadata count and text count are inconsistent"
            )

    @staticmethod
    def _matches(metadata, filters):
        if not filters:
            return True
        return all(metadata.get(key) == value for key, value in filters.items())

    @staticmethod
    def _atomic_pickle(path, value):
        temp_file = path + ".tmp"
        try:
            with open(temp_file, "wb") as target:
                pickle.dump(value, target)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temp_file, path)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
