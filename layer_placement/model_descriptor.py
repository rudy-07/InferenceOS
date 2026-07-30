"""
model_descriptor.py
--------------------
Structured input types for the Phase 3 Layer Placement Engine.

Describes a full transformer model in terms of per-layer weight footprints,
compute cost estimates, and KV cache growth — so the optimizer can reason
about each layer independently rather than treating the model as a monolith.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Quantization type → bits-per-weight (bpw) lookup table.
# Mirrors QUANT_BPW from orchestrator/gguf_selector.py for consistency.
QUANT_BPW: Dict[str, float] = {
    "Q2_K":    2.63,
    "Q3_K_S":  3.0,
    "Q3_K_M":  3.35,
    "Q3_K_L":  3.6,
    "Q4_0":    4.5,
    "Q4_1":    4.75,
    "Q4_K_S":  4.37,
    "Q4_K_M":  4.5,
    "Q4_K_L":  4.9,
    "Q5_0":    5.5,
    "Q5_1":    5.75,
    "Q5_K_S":  5.5,
    "Q5_K_M":  5.68,
    "Q6_K":    6.57,
    "Q8_0":    8.5,
    "F16":     16.0,
    "BF16":    16.0,
    "F32":     32.0,
}

# Layer types we recognize in transformer architectures.
LAYER_TYPE_TRANSFORMER = "transformer"
LAYER_TYPE_EMBEDDING = "embedding"
LAYER_TYPE_LM_HEAD = "lm_head"
LAYER_TYPE_NORM = "norm"


@dataclass
class LayerDescriptor:
    """
    Per-layer descriptor capturing memory and compute characteristics.

    Attributes
    ----------
    layer_index : int
        Zero-based layer index within the model.
    layer_type : str
        Functional type: "transformer", "embedding", "lm_head", "norm".
    size_bytes : int
        Estimated weight memory footprint of this layer in bytes.
    compute_flops : float
        Estimated floating-point operations for one forward-pass token (GFLOPs).
    is_shared : bool
        True for layers that are *not* repeated per block (e.g. embedding, lm_head).
        Shared layers are placed holistically rather than split.
    kv_cache_bytes_per_token : int
        Per-layer KV cache growth per new token (bytes).
        Zero for non-transformer layers.
    """
    layer_index: int
    layer_type: str
    size_bytes: int
    compute_flops: float
    is_shared: bool
    kv_cache_bytes_per_token: int


@dataclass
class ModelDescriptor:
    """
    Full model descriptor used as input to the placement optimizer.

    Can be constructed manually or via :meth:`from_gguf_metadata` using
    the output of ``orchestrator.gguf_parser.read_gguf_metadata()``.

    Attributes
    ----------
    name : str
        Human-readable model name or filename stem.
    architecture : str
        Architecture family string from GGUF metadata (e.g. "llama", "mistral").
    num_layers : int
        Number of transformer blocks (not counting embedding/lm_head).
    hidden_size : int
        Embedding dimension (d_model).
    num_heads : int
        Number of query attention heads.
    num_kv_heads : int
        Number of key/value attention heads (may differ for GQA/MQA).
    head_dim : int
        Dimension per attention head (hidden_size // num_heads).
    vocab_size : int
        Vocabulary size (controls embedding/lm_head tensor size).
    max_context_length : int
        Maximum supported context window from GGUF metadata.
    quant_type : str
        Quantization type string (e.g. "Q4_K_M").
    quant_bpw : float
        Bits-per-weight for this quantization scheme.
    model_size_bytes : int
        Total model file size in bytes (used to distribute weight memory).
    layers : List[LayerDescriptor]
        Ordered list of all layers (embedding + transformer blocks + lm_head).
    """
    name: str
    architecture: str
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    max_context_length: int
    quant_type: str
    quant_bpw: float
    model_size_bytes: int
    layers: List[LayerDescriptor] = field(default_factory=list)

    # ---------------------------------------------------------------------------
    # Factory method: from_gguf_metadata
    # ---------------------------------------------------------------------------

    @classmethod
    def from_gguf_metadata(
        cls,
        metadata: Dict[str, Any],
        model_size_bytes: int,
        quant_type: str = "Q4_K_M",
        model_name: str = "unknown",
    ) -> "ModelDescriptor":
        """
        Construct a :class:`ModelDescriptor` from ``read_gguf_metadata()`` output.

        Parameters
        ----------
        metadata : dict
            Output of ``orchestrator.gguf_parser.read_gguf_metadata()``, containing
            keys: ``arch``, ``num_layers``, ``hidden_size``, ``num_heads``,
            ``num_kv_heads``, ``max_context_length``.
        model_size_bytes : int
            Total model file size in bytes (from ``Path.stat().st_size``).
        quant_type : str
            Quantization string (e.g. "Q4_K_M"). Used for bpw lookup.
        model_name : str
            Human-readable name (e.g. GGUF filename stem).

        Returns
        -------
        ModelDescriptor
            Fully populated descriptor with per-layer breakdowns.
        """
        arch = metadata.get("arch", "llama")
        num_layers = int(metadata.get("num_layers", 32))
        hidden_size = int(metadata.get("hidden_size", 4096))
        num_heads = int(metadata.get("num_heads", 32))
        num_kv_heads = int(metadata.get("num_kv_heads", num_heads))
        max_context_length = int(metadata.get("max_context_length", 4096))
        vocab_size = int(metadata.get("vocab_size", 32000))

        head_dim = hidden_size // max(num_heads, 1)
        quant_bpw = QUANT_BPW.get(quant_type.upper(), 4.5)

        # Build layer descriptors
        layers = _build_layer_descriptors(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            model_size_bytes=model_size_bytes,
            quant_bpw=quant_bpw,
        )

        return cls(
            name=model_name,
            architecture=arch,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            max_context_length=max_context_length,
            quant_type=quant_type,
            quant_bpw=quant_bpw,
            model_size_bytes=model_size_bytes,
            layers=layers,
        )

    # ---------------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------------

    def total_weight_bytes(self) -> int:
        """Sum of weight bytes across all layers."""
        return sum(l.size_bytes for l in self.layers)

    def transformer_layers(self) -> List[LayerDescriptor]:
        """Return only the repeated transformer block layers."""
        return [l for l in self.layers if l.layer_type == LAYER_TYPE_TRANSFORMER]

    def non_transformer_layers(self) -> List[LayerDescriptor]:
        """Return embedding / lm_head / norm layers."""
        return [l for l in self.layers if l.layer_type != LAYER_TYPE_TRANSFORMER]

    def kv_cache_bytes_per_token(self, num_layers: Optional[int] = None) -> int:
        """
        Total KV cache growth per token across all (or a subset of) transformer layers.

        Parameters
        ----------
        num_layers : int, optional
            If given, only count this many transformer layers (for split-plan estimates).
        """
        transformer = self.transformer_layers()
        if num_layers is not None:
            transformer = transformer[:num_layers]
        return sum(l.kv_cache_bytes_per_token for l in transformer)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize descriptor to a JSON-friendly dict."""
        return {
            "name": self.name,
            "architecture": self.architecture,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "max_context_length": self.max_context_length,
            "quant_type": self.quant_type,
            "quant_bpw": self.quant_bpw,
            "model_size_bytes": self.model_size_bytes,
            "model_size_gb": round(self.model_size_bytes / (1024 ** 3), 3),
            "total_layers_including_shared": len(self.layers),
        }


# ---------------------------------------------------------------------------
# Internal builder
# ---------------------------------------------------------------------------

def _build_layer_descriptors(
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    vocab_size: int,
    model_size_bytes: int,
    quant_bpw: float,
) -> List[LayerDescriptor]:
    """
    Decompose a model into per-layer :class:`LayerDescriptor` objects.

    Weight distribution strategy
    ----------------------------
    - **Embedding layer** (index 0): vocab_size × hidden_size × (bpw/8) bytes.
    - **LM Head** (last index): same as embedding (often weight-tied; counted once).
    - **Transformer blocks**: remaining bytes distributed uniformly per block.

    Compute estimates
    -----------------
    For each transformer block (one token forward pass):
      FLOPs ≈ 2 × hidden_size² × (4 + 2 × num_kv_heads / num_heads × head_dim / hidden_size)
    (attention + FFN combined rough estimate)

    KV cache growth per token per layer:
      bytes = 2 × num_kv_heads × head_dim × 2 (fp16)
    """
    descriptors: List[LayerDescriptor] = []
    bytes_per_param = quant_bpw / 8.0

    # Embedding size estimate (quantized)
    embedding_size_bytes = int(vocab_size * hidden_size * bytes_per_param)
    # Cap embedding at 15% of model to avoid over-attribution on large vocabs
    embedding_size_bytes = min(embedding_size_bytes, int(model_size_bytes * 0.15))

    # Remaining bytes after embedding + lm_head (which shares embedding weights)
    total_shared_bytes = embedding_size_bytes  # lm_head is weight-tied → counted once
    transformer_total_bytes = max(0, model_size_bytes - total_shared_bytes)
    bytes_per_transformer_layer = transformer_total_bytes // max(num_layers, 1)

    # Compute estimates
    # Rough FLOPs per transformer block per token (attention + FFN)
    kv_ratio = num_kv_heads / max(num_heads, 1)
    attn_flops = 4.0 * hidden_size * hidden_size / 1e9  # Q/K/V/O projections (GFLOPs)
    ffn_flops = 8.0 * hidden_size * hidden_size / 1e9   # 3-layer FFN estimate (GFLOPs)
    transformer_flops_per_token = attn_flops + ffn_flops

    # KV cache growth: 2 (K+V) × num_kv_heads × head_dim × 2 bytes (fp16)
    kv_bytes_per_token_per_layer = 2 * num_kv_heads * head_dim * 2

    layer_idx = 0

    # Layer 0: Embedding
    descriptors.append(LayerDescriptor(
        layer_index=layer_idx,
        layer_type=LAYER_TYPE_EMBEDDING,
        size_bytes=embedding_size_bytes,
        compute_flops=0.05,   # Trivial lookup
        is_shared=True,
        kv_cache_bytes_per_token=0,
    ))
    layer_idx += 1

    # Transformer blocks
    for i in range(num_layers):
        descriptors.append(LayerDescriptor(
            layer_index=layer_idx,
            layer_type=LAYER_TYPE_TRANSFORMER,
            size_bytes=bytes_per_transformer_layer,
            compute_flops=transformer_flops_per_token,
            is_shared=False,
            kv_cache_bytes_per_token=kv_bytes_per_token_per_layer,
        ))
        layer_idx += 1

    # Final layer: LM Head (weight-tied to embedding; very small residual overhead)
    lm_head_bytes = max(0, model_size_bytes - embedding_size_bytes - (bytes_per_transformer_layer * num_layers))
    descriptors.append(LayerDescriptor(
        layer_index=layer_idx,
        layer_type=LAYER_TYPE_LM_HEAD,
        size_bytes=lm_head_bytes,
        compute_flops=0.1,    # Output projection
        is_shared=True,
        kv_cache_bytes_per_token=0,
    ))

    return descriptors


def infer_quant_type_from_filename(filename: str) -> str:
    """
    Attempt to infer quantization type string from a GGUF filename stem.

    Returns the matched quant key (e.g. "Q4_K_M") or "Q4_K_M" as a safe default.
    """
    stem = filename.upper().replace("-", "_").replace(".", "_")
    # Sort by length descending so longer keys (e.g. "Q4_K_M") match before "Q4"
    for key in sorted(QUANT_BPW.keys(), key=len, reverse=True):
        if key.replace("_", "") in stem.replace("_", ""):
            return key
    return "Q4_K_M"
