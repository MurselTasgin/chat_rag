# /Users/murseltasgin/projects/chat_rag/components/knowledgebase/manager.py
"""
Knowledge Base manager with JSON persistence.
Each KB defines: name, chunker config, embedding model, vector db provider/path, retrieval method defaults.
"""
import os
import json
import uuid
from typing import Dict, Any, List, Optional


class KnowledgeBaseManager:
    def __init__(self, store_path: str = "./.knowledge_bases.json"):
        self.store_path = store_path
        self.kbs: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self.kbs = data
            except Exception:
                self.kbs = {}

    def _save(self) -> None:
        tmp = self.store_path + ".tmp"
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.kbs, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.store_path)

    def list(self) -> List[Dict[str, Any]]:
        return [
            {"kb_id": kb_id, **cfg}
            for kb_id, cfg in self.kbs.items()
        ]

    def get(self, kb_id: str) -> Optional[Dict[str, Any]]:
        return self.kbs.get(kb_id)

    def create(
        self,
        name: str,
        *,
        chunker: Dict[str, Any] = None,
        embedding_model_name: str = None,
        vector_db_provider: str = "chroma",
        vector_db_path: str = None,
        retrieval_method: str = "hybrid",
        extra: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        kb_id = str(uuid.uuid4())[:8]
        cfg = {
            "name": name,
            "chunker": chunker or {"type": "SemanticChunker", "params": {}},
            "embedding_model_name": embedding_model_name,
            "vector_db_provider": vector_db_provider,
            "vector_db_path": vector_db_path,
            "retrieval_method": retrieval_method,
            "extra": extra or {}
        }
        self.kbs[kb_id] = cfg
        self._save()
        return {"kb_id": kb_id, **cfg}

    def update(self, kb_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if kb_id not in self.kbs:
            return None
        self.kbs[kb_id].update(updates)
        self._save()
        return {"kb_id": kb_id, **self.kbs[kb_id]}

    def delete(self, kb_id: str) -> bool:
        if kb_id in self.kbs:
            self.kbs.pop(kb_id)
            self._save()
            return True
        return False


