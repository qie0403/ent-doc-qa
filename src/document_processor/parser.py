"""
文档解析模块 - 支持多格式文档批量解析
Document Parser Module - Multi-format Document Batch Parsing

功能说明：
    支持 PDF、Word、PPT、Excel、TXT、Markdown、CSV 等多种格式的文档解析，
    统一输出为标准化的 Document 对象，包含文本内容、元数据、来源信息等。

技术选型：
    - pypdf：PDF 解析，支持文本提取
    - python-docx：Word 文档解析
    - python-pptx：PPT 解析
    - openpyxl/pandas：Excel/CSV 解析
    - beautifulsoup4：HTML 解析
    - 统一输出 LangChain Document 对象，便于后续处理

优化策略：
    - 按文件扩展名自动选择解析器
    - 异常隔离：单个文件解析失败不影响整体批量处理
    - 元数据丰富：记录文件名、页码、段落号等溯源信息
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from langchain_core.documents import Document
from loguru import logger


class DocumentParser:
    """
    多格式文档解析器

    支持格式：
        - PDF (.pdf)
        - Word (.docx, .doc)
        - PowerPoint (.pptx, .ppt)
        - Excel (.xlsx, .xls)
        - CSV (.csv)
        - 文本 (.txt, .md, .markdown)
        - HTML (.html, .htm)
    """

    def __init__(self):
        """初始化文档解析器"""
        self.supported_formats = {
            ".pdf": self._parse_pdf,
            ".docx": self._parse_docx,
            ".doc": self._parse_docx,  # .doc 尝试用 docx 解析（可能失败）
            ".pptx": self._parse_pptx,
            ".ppt": self._parse_pptx,
            ".xlsx": self._parse_xlsx,
            ".xls": self._parse_xlsx,
            ".csv": self._parse_csv,
            ".txt": self._parse_text,
            ".md": self._parse_text,
            ".markdown": self._parse_text,
            ".html": self._parse_html,
            ".htm": self._parse_html,
        }
        logger.info(f"文档解析器初始化完成，支持 {len(self.supported_formats)} 种格式")

    def parse_file(self, file_path: str | Path) -> List[Document]:
        """
        解析单个文件

        Args:
            file_path: 文件路径

        Returns:
            解析后的 Document 列表

        Raises:
            ValueError: 不支持的文件格式
            Exception: 解析失败时抛出
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = file_path.suffix.lower()
        parser_func = self.supported_formats.get(ext)

        if parser_func is None:
            raise ValueError(f"不支持的文件格式: {ext}")

        try:
            documents = parser_func(file_path)
            logger.info(f"成功解析文件: {file_path.name}, 得到 {len(documents)} 个文档片段")
            return documents
        except Exception as e:
            logger.error(f"解析文件失败: {file_path.name}, 错误: {str(e)}")
            raise

    def parse_directory(self, dir_path: str | Path, recursive: bool = True) -> List[Document]:
        """
        批量解析目录下的所有文档

        Args:
            dir_path: 目录路径
            recursive: 是否递归子目录

        Returns:
            所有解析成功的 Document 列表
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"目录不存在: {dir_path}")

        all_documents = []
        failed_files = []

        # 收集所有文件
        pattern = "**/*" if recursive else "*"
        files = [f for f in dir_path.glob(pattern) if f.is_file()]

        logger.info(f"发现 {len(files)} 个文件，开始批量解析...")

        for file_path in files:
            ext = file_path.suffix.lower()
            if ext not in self.supported_formats:
                continue

            try:
                docs = self.parse_file(file_path)
                all_documents.extend(docs)
            except Exception as e:
                failed_files.append((file_path.name, str(e)))
                logger.warning(f"跳过解析失败的文件: {file_path.name}, 原因: {str(e)}")

        logger.info(
            f"批量解析完成: 成功 {len(all_documents)} 个文档片段, "
            f"失败 {len(failed_files)} 个文件"
        )

        if failed_files:
            logger.warning(f"解析失败的文件列表: {[f[0] for f in failed_files]}")

        return all_documents

    def _parse_pdf(self, file_path: Path) -> List[Document]:
        """
        解析 PDF 文件

        Args:
            file_path: PDF 文件路径

        Returns:
            Document 列表，每个 Document 对应一页
        """
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        documents = []

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
                if text and text.strip():
                    doc = Document(
                        page_content=text.strip(),
                        metadata={
                            "source": str(file_path),
                            "filename": file_path.name,
                            "file_type": "pdf",
                            "page": page_num,
                            "total_pages": len(reader.pages),
                        }
                    )
                    documents.append(doc)
            except Exception as e:
                logger.warning(f"PDF 第 {page_num} 页解析失败: {str(e)}")

        return documents

    def _parse_docx(self, file_path: Path) -> List[Document]:
        """
        解析 Word 文档

        Args:
            file_path: Word 文件路径

        Returns:
            Document 列表，按段落划分
        """
        from docx import Document as DocxDocument

        doc = DocxDocument(str(file_path))
        documents = []
        paragraph_num = 0

        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraph_num += 1
                document = Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "file_type": "docx",
                        "paragraph": paragraph_num,
                        "style": para.style.name if para.style else "Normal",
                    }
                )
                documents.append(document)

        # 解析表格内容
        for table_idx, table in enumerate(doc.tables, start=1):
            table_text = []
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells]
                table_text.append(" | ".join(row_text))

            if table_text:
                document = Document(
                    page_content="\n".join(table_text),
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "file_type": "docx",
                        "table_index": table_idx,
                        "content_type": "table",
                    }
                )
                documents.append(document)

        return documents

    def _parse_pptx(self, file_path: Path) -> List[Document]:
        """
        解析 PowerPoint 文档

        Args:
            file_path: PPT 文件路径

        Returns:
            Document 列表，每页一个 Document
        """
        from pptx import Presentation

        prs = Presentation(str(file_path))
        documents = []

        for slide_num, slide in enumerate(prs.slides, start=1):
            slide_text = []
            title = ""

            # 提取标题
            if slide.shapes.title:
                title = slide.shapes.title.text.strip()

            # 提取所有文本
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text.strip())

            if slide_text:
                document = Document(
                    page_content="\n".join(slide_text),
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "file_type": "pptx",
                        "slide": slide_num,
                        "total_slides": len(prs.slides),
                        "title": title,
                    }
                )
                documents.append(document)

        return documents

    def _parse_xlsx(self, file_path: Path) -> List[Document]:
        """
        解析 Excel 文件

        Args:
            file_path: Excel 文件路径

        Returns:
            Document 列表，每个 Sheet 按行处理
        """
        import pandas as pd

        documents = []
        xls = pd.ExcelFile(str(file_path))

        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                if df.empty:
                    continue

                # 将 DataFrame 转换为文本格式
                # 每行作为一个文档片段，包含表头信息
                headers = list(df.columns)

                for row_idx, row in df.iterrows():
                    row_data = []
                    for col in headers:
                        value = row[col]
                        if pd.notna(value):
                            row_data.append(f"{col}: {value}")

                    if row_data:
                        document = Document(
                            page_content="\n".join(row_data),
                            metadata={
                                "source": str(file_path),
                                "filename": file_path.name,
                                "file_type": "xlsx",
                                "sheet": sheet_name,
                                "row": int(row_idx) + 2,  # Excel 行号从 1 开始，表头占 1 行
                                "total_rows": len(df),
                            }
                        )
                        documents.append(document)

            except Exception as e:
                logger.warning(f"Excel Sheet '{sheet_name}' 解析失败: {str(e)}")

        return documents

    def _parse_csv(self, file_path: Path) -> List[Document]:
        """
        解析 CSV 文件

        Args:
            file_path: CSV 文件路径

        Returns:
            Document 列表，每行一个 Document
        """
        import pandas as pd

        documents = []

        try:
            # 尝试多种编码
            for encoding in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
                try:
                    df = pd.read_csv(str(file_path), encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("无法识别文件编码")

            if df.empty:
                return documents

            headers = list(df.columns)

            for row_idx, row in df.iterrows():
                row_data = []
                for col in headers:
                    value = row[col]
                    if pd.notna(value):
                        row_data.append(f"{col}: {value}")

                if row_data:
                    document = Document(
                        page_content="\n".join(row_data),
                        metadata={
                            "source": str(file_path),
                            "filename": file_path.name,
                            "file_type": "csv",
                            "row": int(row_idx) + 2,
                            "total_rows": len(df),
                        }
                    )
                    documents.append(document)

        except Exception as e:
            logger.error(f"CSV 解析失败: {str(e)}")
            raise

        return documents

    def _parse_text(self, file_path: Path) -> List[Document]:
        """
        解析纯文本文件（.txt, .md, .markdown）

        Args:
            file_path: 文本文件路径

        Returns:
            Document 列表（整个文件作为一个 Document）
        """
        # 尝试多种编码
        content = None
        for encoding in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            raise ValueError("无法识别文件编码")

        file_type = "markdown" if file_path.suffix.lower() in ['.md', '.markdown'] else "text"

        document = Document(
            page_content=content,
            metadata={
                "source": str(file_path),
                "filename": file_path.name,
                "file_type": file_type,
                "char_count": len(content),
            }
        )

        return [document]

    def _parse_html(self, file_path: Path) -> List[Document]:
        """
        解析 HTML 文件

        Args:
            file_path: HTML 文件路径

        Returns:
            Document 列表
        """
        from bs4 import BeautifulSoup

        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'lxml')

        # 移除 script 和 style 标签
        for script in soup(["script", "style"]):
            script.decompose()

        # 提取标题
        title = soup.title.string if soup.title else ""

        # 提取正文文本
        text = soup.get_text()

        # 清理多余空白
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        document = Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "filename": file_path.name,
                "file_type": "html",
                "title": title,
            }
        )

        return [document]

    def get_supported_formats(self) -> List[str]:
        """获取支持的文件格式列表"""
        return list(self.supported_formats.keys())
