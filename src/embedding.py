{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "de25f391",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'src'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[2], line 4\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01msentence_transformers\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m SentenceTransformer\n\u001b[0;32m      3\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mnumpy\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mnp\u001b[39;00m\n\u001b[1;32m----> 4\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01msrc\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mdata_loader\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m load_all_documents\n\u001b[0;32m      6\u001b[0m \u001b[38;5;66;03m# Simple local chunking function to avoid a hard dependency on langchain's text splitter\u001b[39;00m\n\u001b[0;32m      7\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21msimple_chunk_documents\u001b[39m(documents: List[Any], chunk_size: \u001b[38;5;28mint\u001b[39m \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m1000\u001b[39m, chunk_overlap: \u001b[38;5;28mint\u001b[39m \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m200\u001b[39m) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m List[Any]:\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'src'"
     ]
    }
   ],
   "source": [
    "from typing import List, Any\n",
    "from sentence_transformers import SentenceTransformer\n",
    "import numpy as np\n",
    "from src.data_loader import load_all_documents\n",
    "\n",
    "# Simple local chunking function to avoid a hard dependency on langchain's text splitter\n",
    "def simple_chunk_documents(documents: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Any]:\n",
    "    chunks = []\n",
    "    for doc in documents:\n",
    "        text = getattr(doc, 'page_content', str(doc))\n",
    "        start = 0\n",
    "        text_len = len(text)\n",
    "        while start < text_len:\n",
    "            end = min(start + chunk_size, text_len)\n",
    "            chunk_text = text[start:end]\n",
    "            # lightweight container with the same attribute used later\n",
    "            class Chunk:\n",
    "                def __init__(self, content):\n",
    "                    self.page_content = content\n",
    "            chunks.append(Chunk(chunk_text))\n",
    "            if end == text_len:\n",
    "                break\n",
    "            start = end - chunk_overlap\n",
    "    print(f\"[INFO] Split {len(documents)} documents into {len(chunks)} chunks (simple_chunk_documents).\")\n",
    "    return chunks\n",
    "class EmbeddingPipeline:\n",
    "    def __init__(self, model_name: str = \"all-MiniLM-L6-v2\", chunk_size: int = 1000, chunk_overlap: int = 200):\n",
    "        self.chunk_size = chunk_size\n",
    "        self.chunk_overlap = chunk_overlap\n",
    "        self.model = SentenceTransformer(model_name)\n",
    "        print(f\"[INFO] Loaded embedding model: {model_name}\")\n",
    "\n",
    "    def chunk_documents(self, documents: List[Any]) -> List[Any]:\n",
    "        # use the local simple chunker to avoid depending on langchain.text_splitter\n",
    "        return simple_chunk_documents(documents, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)\n",
    "\n",
    "    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:\n",
    "        texts = [chunk.page_content for chunk in chunks]\n",
    "        print(f\"[INFO] Generating embeddings for {len(texts)} chunks...\")\n",
    "        embeddings = self.model.encode(texts, show_progress_bar=True)\n",
    "        print(f\"[INFO] Embeddings shape: {embeddings.shape}\")\n",
    "        return embeddings\n",
    "\n",
    "# Example usage\n",
    "if __name__ == \"__main__\":\n",
    "    docs = load_all_documents(\"data\")\n",
    "    emb_pipe = EmbeddingPipeline()\n",
    "    chunks = emb_pipe.chunk_documents(docs)\n",
    "    embeddings = emb_pipe.embed_chunks(chunks)\n",
    "    print(\"[INFO] Example embedding:\", embeddings[0] if len(embeddings) > 0 else None)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
