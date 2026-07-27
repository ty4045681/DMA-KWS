"""Structured config schema — defaults match Python ``.get(key, default)`` fallbacks."""

from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class PathsConfig:
    librispeech_root: str = ""
    libriphrase100_root: str = ""
    libriphrase460_root: str = ""
    gigaphrase1000_root: str = ""
    processed_root: str = ""
    feature_root: str = ""
    exp_root: str = ""


@dataclass
class TokenizerConfig:
    dict_path: str = ""
    split_with_space: str = " "


@dataclass
class TrainingConfig:
    seed: int = 2025
    recipe: str = ""


@dataclass
class FbankConfig:
    num_mel_bins: int = 80
    frame_length: int = 25
    frame_shift: int = 10
    dither: float = 0.1
    window_type: str = "povey"
    backend: str = "torchaudio_kaldi"
    target_sample_rate: int | None = None
    snip_edges: bool = True
    low_freq: float = 20.0
    high_freq: float = 0.0


@dataclass
class Stage1CheckpointAvgConfig:
    enabled: bool = False
    last_k: int = 10
    pattern: str = "*.ckpt"
    output_name: str = "avg_10.ckpt"


@dataclass
class Stage1ValidationConfig:
    dev_manifest: str = ""
    batch_size: int = 16
    check_val_every_n_epoch: int = 1
    num_decode_batches: int = 0


@dataclass
class Stage1CmvnConfConfig:
    cmvn_file: str = ""
    is_json_cmvn: bool = True


@dataclass
class Stage1StreamConfig:
    """Chunked-attention operating point for the Stage I/II encoder.

    ``chunk_size``/``left_context_frames`` describe the *deployment* operating
    point and must be single values; they are used by validation, offline eval
    and inference, and are never randomized. ``None`` means "not declared" and
    is an error for encoders that support chunking.

    Units are backend specific:
    - ``icefall_zipformer``: frames at 50 Hz (fbank 100 Hz halved by Conv2dSubsampling)
    - ``conformer`` (Wenet): frames at 25 Hz (after the 4x conv2d subsampling)

    ``-1`` means "no chunking" (full context) for both backends.
    """

    chunk_size: int | None = None
    left_context_frames: int | None = None
    train_policy: str = "match"
    train_chunk_size: str = ""
    train_left_context_frames: str = ""


@dataclass
class StreamPolicy:
    """Resolved, validated chunked-attention policy for one encoder."""

    backend: str = "conformer"
    enabled: bool = False
    chunk_size: int = -1
    left_context_frames: int = -1
    train_policy: str = "match"
    train_chunk_sizes: tuple[int, ...] = ()
    train_left_context_frames: tuple[int, ...] = ()

    @property
    def left_context_chunks(self) -> int:
        """Left context expressed in chunks, using icefall's rounding rule."""
        if self.chunk_size <= 0 or self.left_context_frames < 0:
            return -1
        return max(1, self.left_context_frames // self.chunk_size)

    def describe(self) -> str:
        point = f"{self.chunk_size}/{self.left_context_frames}"
        if not self.enabled:
            return f"backend={self.backend} chunking=off (full context)"
        if self.train_policy != "multi":
            train = f"match ({point})"
        elif self.train_chunk_sizes:
            train = (
                f"multi chunk={','.join(map(str, self.train_chunk_sizes))}"
                f" left={','.join(map(str, self.train_left_context_frames))}"
            )
        else:
            train = "multi (backend-native dynamic chunk)"
        return f"backend={self.backend} eval={point} train={train}"


@dataclass
class Stage1Config:
    train_splits: list[str] = field(default_factory=list)
    dev_splits: list[str] = field(default_factory=list)
    sample_rate: int = 16000
    input_dim: int = 80
    encoder_output_dim: int = 144
    attention_heads: int = 4
    linear_units: int = 576
    num_blocks: int = 6
    batch_size_per_gpu: int = 16
    max_epochs: int = 1
    max_train_steps: int = 0
    num_workers: int = 2
    learning_rate: float = 1e-3
    warmup_steps: int = 0
    total_scheduler_steps: int = 0
    dropout_rate: float = 0.1
    positional_dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.0
    ctc_dropout: float = 0.0
    cnn_module_kernel: int = 3
    gradient_clip_val: float = 1.0
    log_interval: int = 10
    checkpoint_dir: str = ""
    log_dir: str = ""
    run_name: str = "stage1_phoneme_ctc"
    fbank_root: str = ""
    audio_root: str = ""
    causal: bool = False
    cnn_module_norm: str = "batch_norm"
    use_dynamic_chunk: bool = False
    use_dynamic_left_chunk: bool = False
    gradient_checkpointing: bool = False
    cmvn: str = ""
    encoder_type: str = "conformer"
    stream: Stage1StreamConfig = field(default_factory=Stage1StreamConfig)
    cmvn_conf: Stage1CmvnConfConfig = field(default_factory=Stage1CmvnConfConfig)
    checkpoint_avg: Stage1CheckpointAvgConfig = field(default_factory=Stage1CheckpointAvgConfig)
    validation: Stage1ValidationConfig = field(default_factory=Stage1ValidationConfig)


@dataclass
class AdapterTrunkConfig:
    """Shape of the trainable trunk that sits on the frozen encoder.

    ``type`` selects how much temporal context the trunk can use:
    ``linear``/``mlp`` are pointwise probes kept for the representation-quality
    control experiments, while ``conv``/``conformer`` can move evidence across
    frames, which is what a transducer-trained encoder needs (RNN-T emission is
    systematically delayed relative to the acoustics).
    """

    type: str = "conv"
    output_dim: int = 192
    num_layers: int = 2
    kernel_size: int = 7
    dropout: float = 0.1
    #: ``conformer`` trunk only, in trunk-input frames (the encoder's own output
    #: rate, 25 Hz for the icefall KWS Zipformer). Required when the encoder is
    #: causal: leaving attention unrestricted there would give the trunk
    #: unlimited lookahead that the deployed pipeline cannot provide.
    chunk_size: int = 0
    #: ``conformer`` trunk only. <0 means all left chunks.
    left_context_chunks: int = -1
    #: ``conformer`` trunk only.
    attention_heads: int = 4
    linear_units: int = 512


@dataclass
class PhonemeAdapterValidationConfig:
    val_check_interval: int = 2000
    num_decode_batches: int = 20
    batch_size: int = 32


@dataclass
class Stage2DataloaderConfig:
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4


@dataclass
class Stage2EmaConfig:
    enabled: bool = False
    decay: float = 0.999
    start_step: int = 0


@dataclass
class Stage2ConsoleConfig:
    rich: bool = True
    device_stats: bool = False
    throughput: bool = False


@dataclass
class Stage2WandbConfig:
    project: str = "dma-kws"
    mode: str = "online"


@dataclass
class Stage2TrackioConfig:
    project: str = "dma-kws"


@dataclass
class Stage2LoggingConfig:
    backends: list[str] = field(default_factory=lambda: ["csv", "tensorboard"])
    wandb: Stage2WandbConfig = field(default_factory=Stage2WandbConfig)
    trackio: Stage2TrackioConfig = field(default_factory=Stage2TrackioConfig)


@dataclass
class Stage2ValidationConfig:
    batch_size: int = 256
    val_check_interval: int = 1000


@dataclass
class Stage2EvalFbankConfig:
    num_mel_bins: int = 80
    frame_length: int = 25
    frame_shift: int = 10
    dither: float = 0.1
    window_type: str = "povey"
    backend: str = "torchaudio_kaldi"
    target_sample_rate: int | None = None
    snip_edges: bool = True
    low_freq: float = 20.0
    high_freq: float = 0.0


@dataclass
class Stage2EvalConfig:
    test_dir: str = ""
    fbank_dir: str = ""
    split: str = "hard"
    aggregate_csv: str = "evaluation_set/test_all_phrase.csv"
    batch_size: int = 256
    num_workers: int = 4
    csv_files: list[str] = field(default_factory=list)
    fbank: Stage2EvalFbankConfig | None = None


@dataclass
class Stage2CheckpointConfig:
    every_n_train_steps: int = 1000
    save_top_k: int = -1
    init_filename: str = "step_{step:06d}"
    finetune_filename: str = "step_{step:06d}_auc_{val_auc:.6f}"


@dataclass
class Stage2GradientDiagnosticsConfig:
    enabled: bool = False
    max_steps: int = 5


@dataclass
class Stage2PrepConfig:
    data_root: str = ""
    output_subdir: str = ""


@dataclass
class PhonemeAdapterConfig:
    """Step A: train a phoneme-CTC trunk on top of a frozen encoder."""

    trunk: AdapterTrunkConfig = field(default_factory=AdapterTrunkConfig)
    ctc_dropout: float = 0.0
    #: Concatenate the CTC log-posteriors onto the trunk output handed to Stage II.
    #: Off by default: the 71-dim posteriorgram is a hard information bottleneck
    #: and its blank-dominated peaks are a poor dense sequence representation.
    expose_posterior: bool = False
    #: Icefall (or Stage I) checkpoint the frozen encoder is initialized from.
    init_checkpoint: str = ""
    train_manifest: str = ""
    dev_manifest: str = ""
    batch_size_per_gpu: int = 32
    num_workers: int = 4
    max_steps: int = 60000
    learning_rate: float = 1e-3
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    warmup_steps: int = 500
    total_scheduler_steps: int = 60000
    accumulate_grad_batches: int = 1
    precision: str = ""
    strategy: str = "auto"
    find_unused_parameters: bool = False
    gradient_clip_val: float = 5.0
    log_interval: int = 50
    val_check_interval: int = 2000
    checkpoint_dir: str = ""
    log_dir: str = ""
    run_name: str = "phoneme_adapter_ctc"
    dataloader: Stage2DataloaderConfig = field(default_factory=Stage2DataloaderConfig)
    validation: PhonemeAdapterValidationConfig = field(
        default_factory=PhonemeAdapterValidationConfig
    )
    logging: Stage2LoggingConfig = field(default_factory=Stage2LoggingConfig)
    checkpoint: Stage2CheckpointConfig = field(default_factory=Stage2CheckpointConfig)


@dataclass
class Stage2PhonemeAdapterConfig:
    """Stage II wiring for the Step A trunk.

    Disabled by default so existing checkpoints keep loading and the pre-adapter
    baseline stays reproducible.
    """

    enabled: bool = False
    #: Step A ``.pt`` used to initialize the trunk when the Stage II checkpoint
    #: does not already carry ``adapter.*`` weights.
    init_checkpoint: str = ""
    #: B1 (frozen trunk) vs B2 (trunk trained jointly with an auxiliary CTC loss).
    freeze: bool = False
    #: Weight of the auxiliary CTC loss in Stage II. 0.0 reproduces the
    #: pre-adapter loss exactly.
    ctc_weight: float = 0.0
    trunk: AdapterTrunkConfig = field(default_factory=AdapterTrunkConfig)
    ctc_dropout: float = 0.0
    expose_posterior: bool = False


@dataclass
class Stage2Config:
    encoder_output_dim: int = 144
    qbyt_embed_dim: int = 128
    qbyt_layers: int = 2
    init_checkpoint: str = ""
    resume_checkpoint: str = ""
    parquet_file: str = ""
    wav_dir: str = ""
    negative_ratio: int = 1
    hard_negative_ratio: int = 1
    sample_lens: int = 5000
    batch_size_per_gpu: int = 64
    max_steps: int = 50000
    num_workers: int = 2
    learning_rate: float = 1e-3
    warmup_steps: int = 2500
    total_scheduler_steps: int = 50000
    accumulate_grad_batches: int = 1
    precision: str = ""
    optimizer: str = "adam"
    weight_decay: float = 0.0
    strategy: str = "auto"
    find_unused_parameters: bool = False
    gradient_clip_val: float = 1.0
    log_interval: int = 10
    val_check_interval: int = 1000
    freeze_encoder: bool = False
    #: Accept QbyT weights trained against the pre-fix pooled readout. Only the
    #: encoder/adapter weights of such a checkpoint are meaningful, so this is a
    #: warm-start escape hatch, not a compatibility mode: scores produced from it
    #: are not comparable with the run it came from.
    allow_legacy_qbyt_readout: bool = False
    checkpoint_dir: str = ""
    log_dir: str = ""
    run_name: str = "stage2_qbyt"
    dataloader: Stage2DataloaderConfig = field(default_factory=Stage2DataloaderConfig)
    ema: Stage2EmaConfig = field(default_factory=Stage2EmaConfig)
    console: Stage2ConsoleConfig = field(default_factory=Stage2ConsoleConfig)
    logging: Stage2LoggingConfig = field(default_factory=Stage2LoggingConfig)
    validation: Stage2ValidationConfig = field(default_factory=Stage2ValidationConfig)
    eval: Stage2EvalConfig = field(default_factory=Stage2EvalConfig)
    checkpoint: Stage2CheckpointConfig = field(default_factory=Stage2CheckpointConfig)
    gradient_diagnostics: Stage2GradientDiagnosticsConfig = field(default_factory=Stage2GradientDiagnosticsConfig)
    prep: Stage2PrepConfig = field(default_factory=Stage2PrepConfig)
    phoneme_adapter: Stage2PhonemeAdapterConfig = field(
        default_factory=Stage2PhonemeAdapterConfig
    )


@dataclass
class WekwsWenetLocatorConfig:
    config: str = ""
    checkpoint: str = ""
    symbol_table: str = ""
    cmvn: str = ""
    bpe_model: str = ""
    threshold: float = 0.0
    min_frames: int = 5
    max_frames: int = 250
    chunk_seconds: float = 0.3
    decoding_chunk_size: int = 16
    score_beam_size: int = 3
    path_beam_size: int = 20
    gpu: int = -1


@dataclass
class SherpaKwsLocatorConfig:
    tokens: str = ""
    encoder: str = ""
    decoder: str = ""
    joiner: str = ""
    keywords_threshold: float = 0.25
    modeling_unit: str = "cjkchar"
    tail_padding_sec: float = 0.66
    provider: str = "cpu"
    num_threads: int = 2


@dataclass
class IcefallPtLocatorConfig:
    root: str = ""
    decode_script: str = ""
    checkpoint: str = ""
    decode_args: list[str] = field(default_factory=list)


@dataclass
class LocatorConfig:
    type: str = "phoneme_ctc"
    frame_shift_sec: float = 0.04
    tokens: str = ""
    encoder: str = ""
    decoder: str = ""
    joiner: str = ""
    keywords_threshold: float = 0.25
    modeling_unit: str = "cjkchar"
    tail_padding_sec: float = 0.66
    provider: str = "cpu"
    num_threads: int = 2
    wekws: WekwsWenetLocatorConfig = field(default_factory=WekwsWenetLocatorConfig)
    root: str = ""
    decode_script: str = ""
    checkpoint: str = ""
    decode_args: list[str] = field(default_factory=list)


@dataclass
class AdaptSweepConfig:
    enabled: bool = False
    n_trials: int = 20
    lambda_forget: float = 1.0
    lph_subset: int = 2000
    search_mix: bool = False
    single_phase: bool = False
    storage: str = ""
    study_name: str = ""


@dataclass
class AdaptValidationConfig:
    val_check_interval: int = 1000
    limit_val_batches: float | None = None


@dataclass
class AdaptConfig:
    keyword: str = "hey eva"
    slug: str = ""
    phase: str = "tts"
    stage: str = "all"
    data_root: str = ""
    exp_root: str = ""
    rank: int = 16
    alpha: int = 32
    lr: float | None = None
    learning_rate: float = 4e-4
    optimizer: str = "adam"
    weight_decay: float = 0.0
    warmup_steps: int = 100
    max_steps: int = 3000
    mix_ratio: float = 0.5
    sample_lens: int = 3000
    batch_size_per_gpu: int = 64
    val_batch_size: int = 64
    num_workers: int = 2
    val_num_workers: int | None = None
    eval_fraction: float = 0.2
    eval_seed: int = 2025
    init_checkpoint: str = ""
    params_file: str = ""
    lora_targets: list[str] = field(default_factory=lambda: ["in_proj_weight", "out_proj.weight"])
    validation: AdaptValidationConfig = field(default_factory=AdaptValidationConfig)
    sweep: AdaptSweepConfig = field(default_factory=AdaptSweepConfig)


@dataclass
class DemoConfig:
    stage1_candidate_margin_sec: float = 0.15
    qbyt_threshold: float = 0.5
    #: Minimum number of *encoder* frames a candidate must yield to be scored.
    #: The equivalent fbank length is derived per backend, because each encoder
    #: subsamples differently (see ``dma_kws.nn.min_input_frames_for_encoder``).
    min_stage2_encoder_frames: int = 1


@dataclass
class RunConfig:
    device: str = "cuda"
    devices: int = 1
    limit_steps: int = 0
    resume_from: str = ""
    init_checkpoint: str = ""
    resume_checkpoint: str = ""
    train_manifest: str = ""
    dev_manifest: str = ""


@dataclass
class DMAKWSConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    fbank: FbankConfig = field(default_factory=FbankConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    phoneme_adapter: PhonemeAdapterConfig = field(default_factory=PhonemeAdapterConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)
    run: RunConfig = field(default_factory=RunConfig)
    adapt: AdaptConfig = field(default_factory=AdaptConfig)
    locator: LocatorConfig = field(default_factory=LocatorConfig)


def register_configs() -> None:
    """Register structured configs with Hydra ConfigStore (optional; not used during compose)."""
    cs = ConfigStore.instance()
    if "config" not in cs.repo:
        cs.store(name="config", node=DMAKWSConfig)
