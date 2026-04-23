from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorDevice
from docling.document_converter import PdfFormatOption
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import json

def load_documents_with_docling(pdf_folder: str) -> list:
    """
    Load and parse all PDFs in a folder using Docling.
    Returns a list of dicts with text content and metadata.
    """

    # Build PDF pipeline options (lower DPI, use lightweight layout model)
    pdf_opts = PdfFormatOption(
        pipeline_options=PdfPipelineOptions(
            # common option names to try:
            layout_model_name="docling-project/docling-layout-tiny",
            #images_scale=1.0,
            do_ocr=False,
            generate_page_images=False,
            generate_picture_images=False
        )
    )

    format_options = {
        InputFormat.PDF: pdf_opts
    }

    converter = DocumentConverter(format_options=format_options)
    documents = []
    pdf_paths = list(Path(pdf_folder).glob("*.pdf"))

    print(f"Found {len(pdf_paths)} PDFs to process...")

    for pdf_path in pdf_paths:
        print(f"Processing: {pdf_path.name}")
        
        # Docling converts PDF to structured markdown
        result = converter.convert(str(pdf_path))
        markdown_text = result.document.export_to_markdown()

        documents.append({
            "source": pdf_path.name,
            "content": markdown_text
        })

        print(f"✅ Done: {pdf_path.name} — {len(markdown_text)} characters extracted")

    return documents


def chunk_documents(documents: list) -> list:
    """
    Split documents into overlapping chunks for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = []
    for doc in documents:
        split_texts = splitter.split_text(doc["content"])
        for i, chunk in enumerate(split_texts):
            chunks.append({
                "text": chunk,
                "source": doc["source"],
                "chunk_id": f"{doc['source']}_chunk_{i}"
            })

    print(f"✅ Total chunks created: {len(chunks)}")
    return chunks


if __name__ == "__main__":
    # Run ingestion
    raw_docs = load_documents_with_docling("data/raw_pdfs")
    chunks = chunk_documents(raw_docs)

    # Save chunks to inspect before embedding
    with open("data/chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)
    
    print(f"✅ All chunks saved to data/chunks_preview.json")