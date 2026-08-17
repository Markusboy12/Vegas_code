"""Модуль памяти на основе Qdrant для хранения и извлечения контекста."""

from typing import Optional
import structlog

logger = structlog.get_logger(__name__)


class MemoryLayer:
    """
    Слой памяти для агента на основе векторной БД Qdrant.
    
    Поддерживает:
    - Сохранение фактов о коде и решениях
    - Поиск похожего контекста
    - Извлечение релевантной информации для текущей задачи
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "agent_memory",
        api_key: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.client = None
        self.qdrant_url = qdrant_url
        self.api_key = api_key
        logger.info(f"MemoryLayer initialized, collection={collection_name}")

    def connect(self) -> bool:
        """Подключиться к Qdrant."""
        try:
            from qdrant_client import QdrantClient
            
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.api_key,
            )
            
            # Проверка существования коллекции
            collections = self.client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                self._create_collection()
            
            logger.info("Connected to Qdrant")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to connect to Qdrant: {e}")
            self.client = None
            return False

    def _create_collection(self):
        """Создать коллекцию для хранения памяти."""
        from qdrant_client.models import Distance, VectorParams
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        logger.info(f"Created collection {self.collection_name}")

    def store_fact(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Сохранить факт в памяти.
        
        Args:
            text: Текстовое представление факта
            metadata: Дополнительные метаданные
        """
        if not self.client:
            logger.warning("Qdrant not connected, skipping store")
            return False
        
        try:
            # В реальном использовании здесь будет эмбеддинг через модель
            # Для простоты используем заглушку
            from qdrant_client.models import PointStruct
            
            # Placeholder для вектора (в реальности нужен embedding model)
            vector = [0.0] * 768
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=hash(text) % (2**63),
                        vector=vector,
                        payload={"text": text, **(metadata or {})},
                    )
                ],
            )
            logger.info(f"Stored fact: {text[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store fact: {e}")
            return False

    def search_similar(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        Найти похожие факты в памяти.
        
        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            
        Returns:
            Список найденных фактов с метаданными
        """
        if not self.client:
            logger.warning("Qdrant not connected, returning empty results")
            return []
        
        try:
            # Placeholder для вектора запроса
            query_vector = [0.0] * 768
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
            )
            
            facts = [
                {"text": r.payload.get("text", ""), **r.payload}
                for r in results
            ]
            logger.info(f"Found {len(facts)} similar facts for query: {query[:30]}...")
            return facts
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def retrieve_context(self, task_description: str) -> str:
        """
        Извлечь релевантный контекст для задачи.
        
        Args:
            task_description: Описание текущей задачи
            
        Returns:
            Строка с контекстом для передачи в LLM
        """
        similar = self.search_similar(task_description, limit=5)
        
        if not similar:
            return ""
        
        context_parts = [f["text"] for f in similar if f.get("text")]
        context = "\n\n".join(context_parts)
        
        logger.info(f"Retrieved context ({len(context)} chars)")
        return context
