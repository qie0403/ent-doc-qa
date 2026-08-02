"""
增量更新模块 - 基于文件哈希的增量检测与更新
Incremental Update Module - Hash-based Change Detection

功能说明：
    实现文档知识库的增量更新机制，通过文件哈希比对检测新增、修改、删除的文件，
    只处理发生变化的文档，大幅提升知识库更新效率。

技术选型：
    - MD5/SHA256 文件哈希：快速检测文件内容变化
    - JSON 元数据存储：记录已处理文件的状态
    - 状态追踪：记录每个文件的处理时间、哈希、分块数量等

优化策略：
    1. 增量检测：只处理新增或修改的文件
    2. 原子更新：先删除旧版本，再插入新版本
    3. 元数据持久化：重启后可恢复处理状态
    4. 批量处理：支持批量增量更新
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime

from loguru import logger


class IncrementalManager:
    """
    增量更新管理器

    功能：
        - 追踪已处理文件的状态（哈希、处理时间等）
        - 检测文件变化（新增、修改、删除）
        - 管理增量更新流程
    """

    def __init__(self, metadata_path: str | Path):
        """
        初始化增量更新管理器

        Args:
            metadata_path: 元数据文件存储路径
        """
        self.metadata_path = Path(metadata_path)
        self.metadata: Dict[str, dict] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """加载元数据"""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logger.info(f"加载元数据成功: {len(self.metadata)} 个文件记录")
            except Exception as e:
                logger.warning(f"加载元数据失败: {str(e)}，将创建新的元数据文件")
                self.metadata = {}
        else:
            logger.info("元数据文件不存在，将创建新的元数据文件")
            self.metadata = {}

    def _save_metadata(self) -> None:
        """保存元数据"""
        try:
            # 确保目录存在
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"元数据保存成功: {len(self.metadata)} 个文件记录")
        except Exception as e:
            logger.error(f"保存元数据失败: {str(e)}")

    def _calculate_file_hash(self, file_path: str | Path) -> str:
        """
        计算文件的 MD5 哈希值

        Args:
            file_path: 文件路径

        Returns:
            文件的 MD5 哈希值
        """
        file_path = Path(file_path)
        hash_md5 = hashlib.md5()

        with open(file_path, "rb") as f:
            # 分块读取大文件
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)

        return hash_md5.hexdigest()

    def get_file_status(self, file_path: str | Path) -> str:
        """
        获取文件状态

        Args:
            file_path: 文件路径

        Returns:
            文件状态: new/modified/unchanged/deleted
        """
        file_path = Path(file_path)
        file_key = str(file_path.resolve())

        if not file_path.exists():
            if file_key in self.metadata:
                return "deleted"
            return "not_found"

        file_hash = self._calculate_file_hash(file_path)

        if file_key not in self.metadata:
            return "new"

        if self.metadata[file_key]["hash"] != file_hash:
            return "modified"

        return "unchanged"

    def scan_directory(
        self,
        dir_path: str | Path,
        recursive: bool = True,
        supported_formats: Optional[Set[str]] = None,
    ) -> Dict[str, List[str]]:
        """
        扫描目录，检测文件变化

        Args:
            dir_path: 目录路径
            recursive: 是否递归子目录
            supported_formats: 支持的文件格式集合（如 {'.pdf', '.docx'}）

        Returns:
            包含各类文件路径的字典：
            {
                "new": [...],        # 新增文件
                "modified": [...],   # 修改文件
                "unchanged": [...],  # 未变化文件
                "deleted": [...],    # 已删除文件
            }
        """
        dir_path = Path(dir_path)
        result = {
            "new": [],
            "modified": [],
            "unchanged": [],
            "deleted": [],
        }

        # 收集当前目录中的所有文件
        current_files = set()
        pattern = "**/*" if recursive else "*"

        for file_path in dir_path.glob(pattern):
            if not file_path.is_file():
                continue

            # 过滤不支持的格式
            if supported_formats and file_path.suffix.lower() not in supported_formats:
                continue

            file_key = str(file_path.resolve())
            current_files.add(file_key)

            status = self.get_file_status(file_path)
            if status == "new":
                result["new"].append(str(file_path))
            elif status == "modified":
                result["modified"].append(str(file_path))
            elif status == "unchanged":
                result["unchanged"].append(str(file_path))

        # 检测已删除的文件（元数据中有但目录中没有的）
        for file_key in self.metadata:
            # 只检查属于该目录的文件
            if file_key.startswith(str(dir_path.resolve())):
                if file_key not in current_files:
                    result["deleted"].append(file_key)

        logger.info(
            f"目录扫描完成: "
            f"新增={len(result['new'])}, "
            f"修改={len(result['modified'])}, "
            f"未变={len(result['unchanged'])}, "
            f"删除={len(result['deleted'])}"
        )

        return result

    def update_file_metadata(
        self,
        file_path: str | Path,
        chunk_count: int,
        doc_ids: Optional[List[str]] = None,
    ) -> None:
        """
        更新文件元数据（处理完成后调用）

        Args:
            file_path: 文件路径
            chunk_count: 分块数量
            doc_ids: 向量数据库中的文档 ID 列表
        """
        file_path = Path(file_path)
        file_key = str(file_path.resolve())
        file_hash = self._calculate_file_hash(file_path)

        self.metadata[file_key] = {
            "filename": file_path.name,
            "hash": file_hash,
            "size": file_path.stat().st_size,
            "chunk_count": chunk_count,
            "doc_ids": doc_ids or [],
            "processed_at": datetime.now().isoformat(),
            "file_type": file_path.suffix.lower(),
        }

        self._save_metadata()

    def remove_file_metadata(self, file_path: str | Path) -> Optional[dict]:
        """
        删除文件元数据（文件被删除时调用）

        Args:
            file_path: 文件路径

        Returns:
            被删除的元数据（包含 doc_ids 等信息）
        """
        file_key = str(Path(file_path).resolve())

        if file_key in self.metadata:
            metadata = self.metadata.pop(file_key)
            self._save_metadata()
            logger.info(f"删除文件元数据: {metadata['filename']}")
            return metadata

        return None

    def get_file_doc_ids(self, file_path: str | Path) -> List[str]:
        """
        获取文件对应的向量文档 ID 列表

        Args:
            file_path: 文件路径

        Returns:
            文档 ID 列表
        """
        file_key = str(Path(file_path).resolve())

        if file_key in self.metadata:
            return self.metadata[file_key].get("doc_ids", [])

        return []

    def get_all_files(self) -> List[dict]:
        """
        获取所有已处理文件的信息

        Returns:
            文件信息列表
        """
        return list(self.metadata.values())

    def get_file_count(self) -> int:
        """获取已处理文件数量"""
        return len(self.metadata)

    def get_total_chunks(self) -> int:
        """获取总分块数量"""
        return sum(meta.get("chunk_count", 0) for meta in self.metadata.values())

    def clear_all(self) -> None:
        """清空所有元数据"""
        self.metadata = {}
        self._save_metadata()
        logger.info("已清空所有元数据")
