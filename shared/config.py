"""统一配置管理：消除全局状态和硬编码常量。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import torch


@dataclass
class DeviceConfig:
    """设备配置。"""
    device: torch.device = field(default_factory=lambda: torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    ))
    use_amp: bool = field(default_factory=lambda: torch.cuda.is_available())

    def __post_init__(self):
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")


@dataclass
class ModelConfig:
    """模型配置。"""
    model_path: Path = field(default_factory=lambda:
        Path(__file__).resolve().parent.parent / "base-models" / "qwen-3-0.6b"
    )
    hidden_size: int = 1024
    style_dim: int = 128
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass
class DataConfig:
    """数据配置。"""
    db_path: Path = field(default_factory=lambda:
        Path(__file__).resolve().parent.parent / "genshin" / "genshin.db"
    )
    cache_dir: Path = field(default_factory=lambda:
        Path(__file__).resolve().parent.parent / "lora" / "cache"
    )
    core_cache_dir: Path = field(default_factory=lambda:
        Path(__file__).resolve().parent.parent / "lora" / "cache_core"
    )
    seed: int = 42
    max_per_speaker: int = 200
    min_sentences: int = 100
    max_len: int = 128
    train_split: float = 0.85  # 说话人级 split
    val_split: float = 0.80    # 训练数据内 split

    # 核心说话人列表
    core_speakers: list[str] = field(default_factory=lambda: [
        "娜维娅", "纳西妲", "温迪", "阿贝多", "茜特菈莉", "八重神子", "玛拉妮", "芙宁娜",
        "玛薇卡", "赛诺", "艾尔海森", "林尼", "提纳里", "荒泷一斗", "那维莱特", "钟离",
        "枫原万叶", "迪希雅", "宵宫", "恰斯卡", "希诺宁", "胡桃", "莫娜", "莱欧斯利",
        "夜兰", "奈芙尔", "凯亚", "神里绫华", "妮露", "刻晴", "柯莱", "菈乌玛",
        "珊瑚宫心海", "安柏", "琴", "「少女」", "可莉", "香菱", "卡维", "卡齐娜",
        "欧洛伦", "基尼奇", "砂糖", "烟绯", "夏洛蒂", "魈", "雅珂达", "菲林斯",
    ])


@dataclass
class TrainConfig:
    """训练配置。"""
    epochs: int = 20
    lr: float = 2e-4
    batch_size: Optional[int] = None  # None 表示自动计算
    grad_accum: int = 1
    num_workers: int = 4
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0

    # LoRA 特定配置
    use_grad_ckpt: bool = False
    fusion_layers: Optional[list[int]] = None
    use_attn_pool: bool = False

    # PK sampler 配置
    pk_p: Optional[int] = None
    pk_k: Optional[int] = None

    # 数据集选择
    use_cache: bool = True
    use_core: bool = False


@dataclass
class EvalConfig:
    """评估配置。"""
    epochs: int = 300
    lr: float = 1e-3
    batch_size: int = 256


@dataclass
class Config:
    """全局配置对象。"""
    device: DeviceConfig = field(default_factory=DeviceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_args(cls, args) -> "Config":
        """从命令行参数创建配置。"""
        config = cls()

        # 更新训练配置
        if hasattr(args, 'rank'):
            config.model.lora_r = args.rank
        if hasattr(args, 'alpha'):
            config.model.lora_alpha = args.alpha
        if hasattr(args, 'batch') and args.batch is not None:
            config.train.batch_size = args.batch
        if hasattr(args, 'grad') and args.grad is not None:
            config.train.grad_accum = args.grad
        if hasattr(args, 'workers') and args.workers is not None:
            config.train.num_workers = args.workers
        if hasattr(args, 'no_cache'):
            config.train.use_cache = not args.no_cache
        if hasattr(args, 'core'):
            config.train.use_core = args.core
        if hasattr(args, 'pk') and args.pk is not None:
            config.train.pk_p, config.train.pk_k = args.pk
        if hasattr(args, 'fusion_layers') and args.fusion_layers is not None:
            config.train.fusion_layers = args.fusion_layers
        if hasattr(args, 'attn_pool'):
            config.train.use_attn_pool = args.attn_pool

        return config

    def auto_batch_size(self) -> int:
        """根据设备自动计算批大小。"""
        if self.train.batch_size is not None:
            return self.train.batch_size

        if self.device.device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info()
            batch = max(1, int(free_bytes / 1e9))
        else:
            import psutil
            batch = max(1, int(psutil.virtual_memory().available / 1e9))

        return (batch // 8) * 8 or 1
