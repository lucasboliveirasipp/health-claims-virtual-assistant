import PyPDF2
import re
from typing import List, Dict, Any


def clean_key(text: str) -> str:
    """Limpa strings para uso em IDs."""

    return re.sub(r"[^\w\s-]", "", text).strip().replace(" ", "-")


def reformat_text_to_markdown_table(text_chunk: str) -> str:
    """
    Tenta converter uma seção de texto bruto de tabela (extraída via PyPDF2)
    em formato Markdown legível, usando análise de alinhamento por whitespace.
    """

    lines = text_chunk.strip().split("\n")
    if not lines:
        return ""

    # 1. Encontrar posições de delimitadores de coluna
    max_len = max(len(line) for line in lines)
    space_counts = [0] * max_len

    for line in lines:
        for i in range(max_len):
            if i < len(line) and line[i].isspace():
                space_counts[i] += 1

    # 2. Definir o limite para identificar um delimitador (70% das linhas têm espaço)
    threshold = len(lines) * 0.7
    delimiters = [
        i for i, count in enumerate(space_counts) if count >= threshold and i > 0
    ]

    # Filtra delimitadores muito próximos
    final_delimiters = []
    if delimiters:
        final_delimiters.append(delimiters[0])
        for i in range(1, len(delimiters)):
            if delimiters[i] - delimiters[i - 1] > 5:
                final_delimiters.append(delimiters[i])

    # 3. Reconstruir as linhas
    markdown_lines = []
    for line in lines:
        row_cells = []
        start_index = 0

        for delim in final_delimiters:
            if delim > start_index:
                cell = line[start_index:delim].strip()
                row_cells.append(cell)
                start_index = delim

        remaining_cell = line[start_index:].strip()
        row_cells.append(remaining_cell)
        cells_content = [cell for cell in row_cells if cell]

        if cells_content:
            markdown_lines.append("| " + " | ".join(cells_content) + " |")

    # 4. Adicionar a linha de separação Markdown

    if len(markdown_lines) > 1:
        num_cols = len(markdown_lines[0].split("|")) - 2
        separator = "|---" * num_cols + "|"
        markdown_lines.insert(1, separator)

    return "\n".join(markdown_lines)


def extract_pages_from_pdf_simple_and_format_tables(
    pdf_file: str, original_name: str
) -> List[Dict[str, Any]]:
    """
    Extrai texto via PyPDF2 e aplica a reconstrução de tabelas por Markdown.
    """

    pages: List[Dict[str, Any]] = []

    try:
        with open(pdf_file, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(reader.pages):
                raw_text = page.extract_text()
                if raw_text and raw_text.strip():
                    formatted_markdown = reformat_text_to_markdown_table(raw_text)
                    # Critério de Decisão: Usa Markdown se a formatação foi bem-sucedida

                    if (
                        formatted_markdown.count("|---|") > 0
                        and formatted_markdown.count("\n") > 5
                    ):

                        final_content = (
                            f"DOCUMENTO: {original_name} | PÁGINA: {i + 1}\n\n"
                            f"### TABELA DE ALTA PRECISÃO\n"
                            f"{formatted_markdown}"
                        )

                        document_type = "TABELA_FORMATADA_MD"

                    else:

                        final_content = (
                            f"DOCUMENTO: {original_name} | PÁGINA: {i + 1}\n\n"
                            f"{raw_text}"
                        )

                        document_type = "TEXTO_BRUTO_PDF"

                    pages.append(
                        {
                            "text": final_content,
                            "metadata": {
                                "document_name": original_name,
                                "page_number": i + 1,
                                "document_type": document_type,
                            },
                        }
                    )

    except Exception as e:
        print(f"Erro ao extrair páginas com PyPDF2 e formatar tabelas: {e}")

    return pages
