import os, traceback
from pathlib import Path
from typing import List, Dict, Set, Generator, Tuple
from codeASTSplitter import CodeASTSplitter


class CodebaseLoader:
    # 默认支持的代码后缀与对应 tree-sitter 解析语言
    DEFAULT_EXT_TO_LANG = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "typescript",
        ".jsx": "typescript",
        ".java": "java",
        # 兜底纯文本/脚本文件（将回退到行切分）
        ".md": "text",
        ".json": "text",
        ".yaml": "text",
        ".yml": "text",
        ".sql": "text",
    }

    # 默认忽略的目录名
    DEFAULT_IGNORE_DIRS: Set[str] = {
        ".git",
        ".svn",
        ".hg",
        ".venv",
        "venv",
        "env",
        ".env",
        "node_modules",
        "bower_components",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".idea",
        ".vscode",
        "dist",
        "build",
        "out",
        "target",
        "bin",
        "obj",
        "qdrant_data",
        "qdrant_hybrid_storage",
    }

    # 默认忽略的文件名或后缀
    DEFAULT_IGNORE_EXTS: Set[str] = {
        ".pyc",
        ".pyo",
        ".pyd",
        ".so",
        ".dll",
        ".dylib",
        ".exe",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".svg",
        ".webp",
        ".zip",
        ".tar",
        ".gz",
        ".7z",
        ".rar",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".min.js",
        ".min.css",
        ".map",
    }

    def __init__(
        self,
        base_dir: str,
        ext_to_lang: Dict[str, str] = None,
        ignore_dirs: Set[str] = None,
        ignore_exts: Set[str] = None,
        max_file_size_kb: int = 500,  # 超过 500KB 的文件通常是打包产物/大数据文件，直接跳过
    ):
        self.base_dir = Path(base_dir).resolve()
        self.ext_to_lang = ext_to_lang or self.DEFAULT_EXT_TO_LANG
        self.ignore_dirs = ignore_dirs or self.DEFAULT_IGNORE_DIRS
        self.ignore_exts = ignore_exts or self.DEFAULT_IGNORE_EXTS
        self.max_file_size_bytes = max_file_size_kb * 1024

    def is_ignored(self, path: Path) -> bool:
        """检查文件或目录是否应该被忽略"""
        # 1. 检查各级目录是否在忽略名单中
        for part in path.parts:
            if part in self.ignore_dirs or (
                part.startswith(".")
                and part not in {".", ".."}
                and part in self.ignore_dirs
            ):
                return True

        # 2. 检查文件后缀
        if path.suffix.lower() in self.ignore_exts:
            return True

        # 3. 如果设置了白名单后缀且当前后缀不在其中
        if self.ext_to_lang and path.suffix.lower() not in self.ext_to_lang:
            return True

        # 4. 检查文件大小
        if path.is_file() and path.stat().st_size > self.max_file_size_bytes:
            return True

        return False

    def walk_files(self) -> Generator[Path, None, None]:
        """递归遍历目录，产出有效代码文件路径"""
        for root, dirs, files in os.walk(self.base_dir, topdown=True):
            root_path = Path(root)

            # 原地修改 dirs 以跳过忽略的目录，避免无谓的递归下钻
            dirs[:] = [
                d
                for d in dirs
                if d not in self.ignore_dirs and not d.startswith(".git")
            ]

            for file_name in files:
                file_path = root_path / file_name
                if not self.is_ignored(file_path):
                    yield file_path

    def load_and_split_all(
        self, max_chunk_size: int = 1500
    ) -> Generator[Tuple[str, List[Dict]], None, None]:
        """遍历整个目录并使用 AST 切分所有文件，逐个文件（页面）产出 (file_path, chunks)"""
        # 缓存不同语言的 AST 切分器实例
        splitters = {}

        for file_path in self.walk_files():
            # 获取相对于 base_dir 的相对路径，方便作为唯一标识和存储展示
            rel_path = file_path.relative_to(self.base_dir).as_posix()
            lang = self.ext_to_lang.get(file_path.suffix.lower())

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if not content.strip():
                    continue

                # 获取或创建该语言的 Splitter
                if lang not in splitters:
                    splitters[lang] = CodeASTSplitter(
                        language=lang, max_chunk_size=max_chunk_size
                    )

                chunks = splitters[lang].split_file(
                    file_path=rel_path, code_content=content
                )
                if chunks:
                    chunk_dicts = [
                        c.__dict__ if hasattr(c, "__dict__") else c for c in chunks
                    ]
                    yield rel_path, chunk_dicts

            except Exception as e:
                print(f"[跳过] 解析文件失败 {rel_path}")
                traceback.print_exc()
