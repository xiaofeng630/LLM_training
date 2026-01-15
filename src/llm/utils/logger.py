import logging
from pathlib import Path
from datetime import datetime

def setup_run_dir(
    experiment: str = "pretraining",
    run_name: str | None = None,
):
    project_root = Path(__file__).resolve().parents[3]
    base_dir = project_root / "logs" / experiment
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = run_name or "run"
    run_dir = base_dir / f"{timestamp}_{run_name}"

    run_dir.mkdir(parents=True, exist_ok=False)

    # 子目录
    (run_dir / "checkpoints").mkdir()
    (run_dir / "samples").mkdir()

    return run_dir


def setup_logger(name: str, log_file: Path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)

    # 控制台
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logger initialized. Log file: {log_file}")
    return logger

