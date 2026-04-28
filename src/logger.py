"""运行日志模块。

每次运行自动在 runlogs/ 下创建带时间戳的目录，保存：
- run.log：完整运行日志（与终端输出一致，日志文件附带时间戳）
- params.txt：命令行参数快照

用法：
    from src.logger import RunLogger

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with RunLogger(args) as log:
        log.section("开始训练")
        train_loop(args)

模块级代理函数（可直接调用，无需持有 RunLogger 实例）：
    from src.logger import info, section, warning, progress
    info("消息")
    section("阶段标题")
    warning("警告信息")
    progress(current, total, key=value, ...)
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# 模块级全局（用于代理函数）
# ---------------------------------------------------------------------------
_active_logger: Optional["RunLogger"] = None


def info(message: str):
    """输出普通信息。"""
    if _active_logger is not None:
        _active_logger.info(message)
    else:
        print(message)


def section(title: str):
    """输出带分隔线的阶段标题。"""
    if _active_logger is not None:
        _active_logger.section(title)
    else:
        print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}\n")


def warning(message: str):
    """输出警告信息。"""
    if _active_logger is not None:
        _active_logger.warning(message)
    else:
        print(f"[警告] {message}")


def progress(current: int, total: int, **metrics):
    """输出进度行（可被后续行覆盖刷新）。

    参数：
        current: 当前步数
        total:   总步数
        **metrics: 额外指标键值对（值应可转换为 float）
    """
    if _active_logger is not None:
        _active_logger.progress(current, total, **metrics)
    else:
        parts = [f"进度 {current}/{total}"]
        for k, v in metrics.items():
            try:
                parts.append(f"{k}: {float(v):.4f}")
            except (TypeError, ValueError):
                parts.append(f"{k}: {v}")
        print(" | ".join(parts), end="               \r")


def get_logger() -> Optional["RunLogger"]:
    """获取当前活跃的 RunLogger 实例（无则返回 None）。"""
    return _active_logger


# ---------------------------------------------------------------------------
# Tee 流：同时写入文件和控制台
# ---------------------------------------------------------------------------
class _TeeStream:
    """Tee 流：将写入内容同时输出到文件和控制台。"""

    def __init__(self, file, stream):
        self.file = file
        self.stream = stream

    def write(self, data):
        self.file.write(data)
        self.stream.write(data)

    def flush(self):
        self.file.flush()
        self.stream.flush()


# ---------------------------------------------------------------------------
# 时间戳格式化
# ---------------------------------------------------------------------------
def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# RunLogger 主类
# ---------------------------------------------------------------------------
class RunLogger:
    """单次运行的日志管理器。

    职责：
    - 在 runlogs/ 下创建带时间戳的运行目录
    - 保存命令行参数到 params.txt
    - 将 stdout/stderr 同时输出到控制台和 run.log（Tee 方式）
    - 提供 section() / info() / warning() / progress() 格式化输出方法
    - 自动记录开始/结束时间、git commit、总耗时
    """

    def __init__(self, args, log_dir: str = "runlogs"):
        # progress_write_interval: 控制写入 run.log 的最小间隔（秒），避免每次 progress 调用都写文件造成 IO 开销
        # 默认 1.0s
        self._progress_write_interval = getattr(args, 'progress_write_interval', 1.0)
        self._start_time = time.time()
        self._args = args
        self._log_dir = log_dir

        # 构建运行目录名
        tag = getattr(args, "tag", "") or ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_{tag}" if tag else timestamp
        self.run_name = run_name
        self.run_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        # 保存参数快照
        self._save_params()

        # 打开日志文件（覆盖写，后续 Tee 追加）
        self._log_path = os.path.join(self.run_dir, "run.log")
        self._log_file = open(self._log_path, "w", encoding="utf-8")

        # 保存原始流并替换为 Tee
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _TeeStream(self._log_file, self._orig_stdout)
        sys.stderr = _TeeStream(self._log_file, self._orig_stderr)

        # 注册为活跃 logger
        global _active_logger
        _active_logger = self

        # 记录上次 progress 写入的时间，用于节流
        self._last_progress_write_time = 0.0

        # 写头部信息
        self._write_header()

    def _save_params(self):
        """将命令行参数保存为可读文本格式。"""
        params = {}
        if hasattr(self._args, "__dict__"):
            params = vars(self._args)
        elif isinstance(self._args, dict):
            params = self._args

        path = os.path.join(self.run_dir, "params.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"运行目录: {self.run_dir}\n")
            f.write("-" * 50 + "\n")
            for k, v in sorted(params.items()):
                f.write(f"  {k}: {v}\n")
            f.write("-" * 50 + "\n")
            # 也保存 JSON 格式便于程序读取
        json_path = os.path.join(self.run_dir, "params.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2, default=str)
        except Exception:
            pass

    def _git_hash(self) -> str:
        """获取当前 git commit hash。"""
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            return "N/A"

    def _write_header(self):
        """在日志头部写入运行概览信息。"""
        header = (
            f"{'=' * 60}\n"
            f"  运行开始: {_timestamp()}\n"
            f"  运行目录: {self.run_dir}\n"
            f"  Git: {self._git_hash()}\n"
            f"{'=' * 60}\n"
        )
        # 直接写入文件（避免在 Tee 中自我循环）
        self._log_file.write(header)
        self._log_file.flush()
        self._orig_stdout.write(header)
        self._orig_stdout.flush()

        # 参数概要输出到控制台
        if hasattr(self._args, "__dict__"):
            params = [(k, v) for k, v in vars(self._args).items()
                      if v is not None and v != "" and v is not False]
            if params:
                for k, v in sorted(params):
                    self._orig_stdout.write(f"    {k}: {v}\n")
                self._orig_stdout.flush()
                self._log_file.write(
                    "".join(f"    {k}: {v}\n" for k, v in sorted(params))
                )
                self._log_file.flush()

    # ------------------------------------------------------------------
    # 输出方法
    # ------------------------------------------------------------------
    def info(self, message: str):
        """输出普通信息。"""
        ts = _timestamp()
        self._log_file.write(f"[{ts}] [信息] {message}\n")
        self._log_file.flush()
        self._orig_stdout.write(f"[信息] {message}\n")
        self._orig_stdout.flush()

    def section(self, title: str):
        """输出带分隔线的阶段标题。"""
        ts = _timestamp()
        block = f"\n{'=' * 60}\n  {title}\n{'=' * 60}\n"
        self._log_file.write(f"[{ts}] ----- {title} -----\n")
        self._log_file.write(block)
        self._log_file.flush()
        self._orig_stdout.write(block)
        self._orig_stdout.flush()

    def warning(self, message: str):
        """输出警告信息。"""
        ts = _timestamp()
        self._log_file.write(f"[{ts}] [警告] {message}\n")
        self._log_file.flush()
        self._orig_stdout.write(f"[警告] {message}\n")
        self._orig_stdout.flush()

    def progress(self, current: int, total: int, **metrics):
        """输出进度行（可被后续行覆盖刷新）。

        参数：
            current: 当前步数
            total:   总步数
            **metrics: 额外指标键值对
        """
        parts = [f"进度 {current}/{total}"]
        for k, v in metrics.items():
            try:
                parts.append(f"{k}: {float(v):.4f}")
            except (TypeError, ValueError):
                parts.append(f"{k}: {v}")
        line = " | ".join(parts)
        padding = " " * max(0, 80 - len(line))

        ts = _timestamp()
        # 写入文件采用节流，减少磁盘 IO：仅在距离上次写入超过阈值时才写入 run.log
        try:
            now = time.time()
            if now - getattr(self, '_last_progress_write_time', 0.0) >= self._progress_write_interval:
                self._log_file.write(f"[{ts}] [进度] {line}\n")
                self._log_file.flush()
                self._last_progress_write_time = now
        except Exception:
            # 日志写入不应影响主流程，失败时打印到原始 stdout
            try:
                self._orig_stdout.write(f"[日志写入错误] {ts}\n")
                self._orig_stdout.flush()
            except Exception:
                pass

        # 控制台仍按每次调用刷新进度行（覆盖式输出）
        try:
            self._orig_stdout.write(f"\r{line}{padding}")
            self._orig_stdout.flush()
        except Exception:
            pass

    def close(self):
        """关闭日志器：输出耗时统计、恢复原始流。"""
        global _active_logger
        if self._log_file is None:
            return

        elapsed = time.time() - self._start_time
        elapsed_str = (
            f"{elapsed:.2f}s"
            if elapsed < 120
            else f"{elapsed:.0f}s ({elapsed / 60:.1f}min)"
        )
        footer = (
            f"\n{'=' * 60}\n"
            f"  运行结束: {_timestamp()}\n"
            f"  总耗时: {elapsed_str}\n"
            f"{'=' * 60}\n"
        )
        # 先写文件
        self._log_file.write(f"[{_timestamp()}] ====== 运行结束 ======\n")
        self._log_file.write(f"[{_timestamp()}] 总耗时: {elapsed_str}\n")
        self._log_file.flush()
        # 再写控制台
        self._orig_stdout.write(footer)
        self._orig_stdout.flush()

        # 恢复原始流
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

        # 关闭文件
        self._log_file.close()
        self._log_file = None

        # 清空全局
        _active_logger = None

    # ------------------------------------------------------------------
    # 上下文管理器支持
    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        # 不吞没异常
        return False
