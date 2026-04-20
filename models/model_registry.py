# models/model_registry.py
"""
Model Registry: Track, version, and manage multiple model variants.
Provides a registry pattern for model lifecycle management.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, List, Any
from enum import Enum

from loguru import logger


class ModelStage(Enum):
    EXPERIMENTAL = "experimental"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    """Metadata for a registered model version."""
    name: str
    version: str
    model_id: str                              # HuggingFace model ID
    stage: ModelStage = ModelStage.EXPERIMENTAL
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    run_id: Optional[str] = None              # MLflow run ID

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stage"] = self.stage.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelVersion":
        d = d.copy()
        d["stage"] = ModelStage(d.get("stage", "experimental"))
        return cls(**d)


class ModelRegistry:
    """
    File-backed model registry for tracking SD model versions.
    Supports promotion through stages: experimental → staging → production.

    Each unique model_id + version is stored with metadata, metrics, and stage.
    """

    def __init__(self, registry_path: str = "./models/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry: Dict[str, Dict[str, ModelVersion]] = {}
        self._load()

    def _load(self):
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path) as f:
                    raw = json.load(f)
                for name, versions in raw.items():
                    self._registry[name] = {
                        v: ModelVersion.from_dict(meta)
                        for v, meta in versions.items()
                    }
                logger.debug(f"Loaded {sum(len(v) for v in self._registry.values())} model versions from registry")
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}. Starting fresh.")
                self._registry = {}

    def _save(self):
        """Persist registry to disk."""
        raw = {}
        for name, versions in self._registry.items():
            raw[name] = {v: mv.to_dict() for v, mv in versions.items()}
        with open(self.registry_path, "w") as f:
            json.dump(raw, f, indent=2)

    def register(
        self,
        name: str,
        version: str,
        model_id: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        stage: ModelStage = ModelStage.EXPERIMENTAL,
        run_id: Optional[str] = None,
    ) -> ModelVersion:
        """Register a new model version."""
        mv = ModelVersion(
            name=name,
            version=version,
            model_id=model_id,
            description=description,
            tags=tags or {},
            metrics=metrics or {},
            stage=stage,
            run_id=run_id,
        )
        if name not in self._registry:
            self._registry[name] = {}
        self._registry[name][version] = mv
        self._save()
        logger.info(f"Registered model: {name}/{version} (stage={stage.value})")
        return mv

    def get_version(self, name: str, version: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        return self._registry.get(name, {}).get(version)

    def get_latest(self, name: str, stage: Optional[ModelStage] = None) -> Optional[ModelVersion]:
        """Get the latest version (by registration time), optionally filtered by stage."""
        versions = self._registry.get(name, {})
        if not versions:
            return None
        candidates = list(versions.values())
        if stage:
            candidates = [v for v in candidates if v.stage == stage]
        if not candidates:
            return None
        return max(candidates, key=lambda v: v.registered_at)

    def get_production_model(self, name: str) -> Optional[ModelVersion]:
        """Get the current production model."""
        return self.get_latest(name, stage=ModelStage.PRODUCTION)

    def promote(self, name: str, version: str, new_stage: ModelStage) -> bool:
        """Promote a model version to a new stage."""
        mv = self.get_version(name, version)
        if not mv:
            logger.warning(f"Model {name}/{version} not found")
            return False
        old_stage = mv.stage
        mv.stage = new_stage
        mv.updated_at = time.time()
        self._save()
        logger.info(f"Promoted {name}/{version}: {old_stage.value} → {new_stage.value}")
        return True

    def update_metrics(self, name: str, version: str, metrics: Dict[str, float]) -> bool:
        """Update performance metrics for a version."""
        mv = self.get_version(name, version)
        if not mv:
            return False
        mv.metrics.update(metrics)
        mv.updated_at = time.time()
        self._save()
        return True

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._registry.keys())

    def list_versions(self, name: str) -> List[ModelVersion]:
        """List all versions for a model, newest first."""
        versions = list(self._registry.get(name, {}).values())
        return sorted(versions, key=lambda v: v.registered_at, reverse=True)

    def delete_version(self, name: str, version: str) -> bool:
        """Delete a specific model version from the registry."""
        if name in self._registry and version in self._registry[name]:
            del self._registry[name][version]
            if not self._registry[name]:
                del self._registry[name]
            self._save()
            logger.info(f"Deleted {name}/{version} from registry")
            return True
        return False

    def to_dataframe(self):
        """Convert registry to pandas DataFrame for display."""
        try:
            import pandas as pd
            rows = []
            for name, versions in self._registry.items():
                for version, mv in versions.items():
                    rows.append({
                        "Name": mv.name,
                        "Version": mv.version,
                        "Stage": mv.stage.value,
                        "Model ID": mv.model_id[:50],
                        "Description": mv.description[:60],
                        "Registered": time.strftime("%Y-%m-%d %H:%M", time.localtime(mv.registered_at)),
                    })
            return pd.DataFrame(rows)
        except ImportError:
            return None

    def initialize_default_model(self):
        """Register the default SD v1.5 model if not present."""
        name = "stable-diffusion"
        if not self.get_version(name, "v1.5"):
            self.register(
                name=name,
                version="v1.5",
                model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
                description="Stable Diffusion v1.5 — base model from RunwayML/CompVis",
                tags={"resolution": "512x512", "framework": "diffusers", "source": "huggingface"},
                stage=ModelStage.PRODUCTION,
            )
            logger.info("Initialized default SD v1.5 in registry")
