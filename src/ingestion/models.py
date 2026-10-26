from typing import Literal

from pydantic import BaseModel, Field


class DocumentSection(BaseModel):
    heading: str | None = None
    text: str = Field(..., min_length=1)
    page: int | None = None


class Document(BaseModel):
    document_id: str
    title: str
    filename: str 
    source: str
    file_type: str
    sections: list[DocumentSection]


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str = Field(..., min_length=1)
    filename: str 
    source: str
    section: str | None = None
    page: int | None = None
    chunk_index: int

