"""YAML config 로더 + dot 접근 헬퍼."""
from pathlib import Path
import yaml


class DotDict(dict):
    """딕셔너리에 .필드 접근을 허용하는 래퍼.

    중첩 dict는 생성 시 모두 DotDict로 변환되어 저장된다.
    이렇게 해야 cfg.train.epochs = 5 같은 중첩 속성 할당이 원본에 반영된다.
    """
    def __init__(self, data=None):
        super().__init__()
        if data is None:
            return
        for k, v in dict(data).items():
            self[k] = DotDict(v) if isinstance(v, dict) else v

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = DotDict(value) if isinstance(value, dict) else value


def load_config(path: str) -> DotDict:
    with Path(path).open("r", encoding="utf-8") as f:
        return DotDict(yaml.safe_load(f))
