from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document

from src.retrieval.intent import infer_policy_type, infer_policy_types

HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")

def extract_title(text: str, fallback: str) -> str:
    """标题提取与分级"""
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line.lstrip("#").strip()
    return Path(fallback).stem


def split_markdown_sections(doc: Document) -> List[Document]:
    """按章节切分"""
    source = doc.metadata.get("source", "")
    title = extract_title(doc.page_content, source)
    policy_type = infer_policy_type(source, title)
    policy_types = infer_policy_types(source, title)
    lines = doc.page_content.splitlines()
    sections: List[Document] = []
    current_lines: List[str] = []
    current_section = title

    def flush() -> None:
        nonlocal current_lines, current_section
        text = "\n".join(current_lines).strip()
        if not text:
            current_lines = []
            return
        metadata = dict(doc.metadata or {})
        metadata.update({
            "doc_title": title,
            "policy_type": policy_type,
            "policy_types": policy_types,
            "section_title": current_section,
            "source": source,
        })
        sections.append(Document(page_content=text, metadata=metadata))
        current_lines = []

    for line in lines:
        stripped = line.strip()
        #匹配是否有标题
        heading = HEADING_RE.match(stripped)
        if heading and current_lines:
            flush()
            current_section = heading.group(2).strip()
        elif heading:
            current_section = heading.group(2).strip()
        current_lines.append(line)
    flush()
   #调用 enrich_document_metadata(doc) 作为兜底，生成一个带基本元数据的文档返回
    return sections or [enrich_document_metadata(doc)]


def enrich_document_metadata(doc: Document) -> Document:
    """元数据富化 每个切出来的小文档打上标签"""
#包括 doc_title（文档名）、section_title（当前章节名）、policy_type（政策类型，通过外部意图推断
    source = doc.metadata.get("source", "")
    title = extract_title(doc.page_content, source)
    metadata = dict(doc.metadata or {})
    metadata.setdefault("doc_title", title)
    metadata.setdefault("section_title", title)
    metadata.setdefault("policy_type", infer_policy_type(source, title))
    metadata.setdefault("policy_types", infer_policy_types(source, title))
    metadata.setdefault("source", source)
    if "page" in metadata and "page_number" not in metadata:
        metadata["page_number"] = metadata["page"]
    return Document(page_content=doc.page_content, metadata=metadata)


def prepare_documents_for_indexing(documents: Iterable[Document]) -> List[Document]:
    """新增的 split_markdown_sections + enrich_document_metadata 的入口。它负责按 # 标题切分，并打上 policy_type 标签和 section_title"""
    prepared: List[Document] = []
    for doc in documents:
        source = str(doc.metadata.get("source", ""))
        if source.lower().endswith((".md", ".txt")):
            prepared.extend(split_markdown_sections(doc))
        else:
            prepared.append(enrich_document_metadata(doc))
    return prepared
