
import base64, inspect, io, os, sys, tempfile, wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

# Works both in a real .py file (Dynabench) and in Colab notebooks
ZIP_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ZIP_ROOT / "ups_challenge_baselines"))

BiMambaMSM = None
BIMAMBA_IMPORT_ERROR: Optional[Exception] = None
CUDA_AVAILABLE = torch.cuda.is_available()
try:
    from scripts.train_mel_msm_bimamba2 import BiMambaMSM  # noqa: E402
except Exception as exc:
    BiMambaMSM = None
    BIMAMBA_IMPORT_ERROR = exc


class ModelController:
    # Dynalab-style wrapper.
    SR = 16000
    N_FFT = 400
    HOP = 160
    N_MELS = 80
    EPS = 1e-6
    VALID_FRAME_THRESHOLD = -13.7
    OUTPUT_DIM = 128

    @staticmethod
    def _decode_wav_bytes_stdlib(wav_bytes: bytes) -> (torch.Tensor, int):
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            num_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            num_frames = wf.getnframes()
            raw = wf.readframes(num_frames)

        if sample_width == 1:
            audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            audio = (audio - 128.0) / 128.0
        elif sample_width == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 3:
            a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            signed = (
                a[:, 0].astype(np.int32)
                | (a[:, 1].astype(np.int32) << 8)
                | (a[:, 2].astype(np.int32) << 16)
            )
            sign = signed & 0x800000
            signed = signed - (sign << 1)
            audio = signed.astype(np.float32) / 8388608.0
        elif sample_width == 4:
            audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

        if num_channels > 1:
            audio = audio.reshape(-1, num_channels).mean(axis=1)
        wav = torch.from_numpy(audio).unsqueeze(0).to(torch.float32)  # [1, N]
        return wav, sample_rate

    @staticmethod
    def _load_checkpoint(ckpt_path: Path) -> Dict[str, Any]:
        """Load checkpoint robustly across torch versions (incl. 2.6 weights_only change)."""
        load_kwargs = {"map_location": "cpu"}
        torch_version_cls = None
        try:
            torch_version_cls = torch.torch_version.TorchVersion
        except Exception:
            torch_version_cls = None

        # Preferred path on newer torch: keep weights_only=True and allowlist TorchVersion metadata.
        try:
            if torch_version_cls is not None and hasattr(torch.serialization, "safe_globals"):
                with torch.serialization.safe_globals([torch_version_cls]):
                    return torch.load(str(ckpt_path), weights_only=True, **load_kwargs)
            return torch.load(str(ckpt_path), weights_only=True, **load_kwargs)
        except Exception:
            # Trusted local checkpoint fallback for environments where weights_only path fails.
            try:
                return torch.load(str(ckpt_path), weights_only=False, **load_kwargs)
            except TypeError:
                # Older torch versions without weights_only argument.
                return torch.load(str(ckpt_path), **load_kwargs)

    def __init__(self, device: str = "cuda" if CUDA_AVAILABLE else "cpu") -> None:
        self.initialized = True
        requested_device = torch.device(device)
        self.model: Optional[torch.nn.Module] = None
        self.backend = "uninitialized"
        self.d_model = 128
        self.num_layers = 1
        self.num_clusters = 256
        self.discrete_mode = True
        self._inference_debug_printed = False
        self._dim_warning_printed = False
        self.allow_fallback = os.getenv("UPS_ALLOW_BIMAMBA_FALLBACK", "0").strip() == "1"

        ckpt_path = ZIP_ROOT / "app" / "resources" / "ckpt_step_11000_infer.pt"
        ckpt_exists = ckpt_path.exists()
        print(f"[ModelController:init] BiMamba import ok: {BiMambaMSM is not None}")
        print(f"[ModelController:init] checkpoint={ckpt_path} exists={ckpt_exists}")
        if not ckpt_exists:
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        ckpt = self._load_checkpoint(ckpt_path)

        cfg = ckpt.get("cfg", {})
        self.d_model = int(cfg.get("d_model", 128))
        self.num_layers = int(cfg.get("num_layers", 1))
        self.num_clusters = int(cfg.get("num_clusters", 256))
        self.discrete_mode = bool(cfg.get("discrete_mode", True))
        self.OUTPUT_DIM = self.d_model
        print(
            "[ModelController:init] cfg d_model="
            f"{self.d_model} num_layers={self.num_layers} num_clusters={self.num_clusters}"
        )
        selected_device = (
            torch.device("cuda")
            if requested_device.type == "cuda" and torch.cuda.is_available()
            else torch.device("cpu")
        )
        selected_backend = "bimamba_cuda" if selected_device.type == "cuda" else "bimamba_cpu"
        if BiMambaMSM is None:
            selected_backend = "bimamba_unavailable"
        print(f"[ModelController:init] selected backend={selected_backend} device={selected_device}")
        if BiMambaMSM is None:
            err = repr(BIMAMBA_IMPORT_ERROR) if BIMAMBA_IMPORT_ERROR is not None else "unknown import error"
            raise RuntimeError(f"BiMambaMSM import failed: {err}")

        state = ckpt.get("model_state", {})
        if not isinstance(state, dict) or not state:
            raise RuntimeError("Checkpoint model_state is missing or empty.")

        self.device = selected_device
        self.model = self._build_bimamba_model().to(self.device)
        self.model = self.model.float()
        if "lid_head.weight" in state and hasattr(self.model, "lid_head"):
            n_lids = int(state["lid_head.weight"].shape[0])
            self.model.lid_head = torch.nn.Linear(self.d_model, n_lids).to(self.device)

        missing_keys, unexpected_keys = self.model.load_state_dict(state, strict=False)
        print(
            "[ModelController:init] load_state_dict strict=False "
            f"missing={len(missing_keys)} unexpected={len(unexpected_keys)}"
        )
        self._validate_checkpoint_compatibility(state, missing_keys, unexpected_keys)
        self.model.eval()
        self.backend = "bimamba_cuda" if self.device.type == "cuda" else "bimamba_cpu"
        print(f"[ModelController:init] backend={self.backend} device={self.device}")

        self.mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.SR,
            n_fft=self.N_FFT,
            hop_length=self.HOP,
            n_mels=self.N_MELS,
            power=2.0,
        )
        self.mel_fn = self.mel_fn.to("cpu")

    def _build_bimamba_model(self) -> torch.nn.Module:
        if BiMambaMSM is None:
            raise RuntimeError("BiMambaMSM is unavailable.")
        try:
            ctor_sig = inspect.signature(BiMambaMSM.__init__)
        except Exception as exc:
            raise RuntimeError(f"Unable to inspect BiMambaMSM constructor: {exc}") from exc
        supported = set(ctor_sig.parameters.keys())
        kwargs: Dict[str, Any] = {}
        if "d_model" in supported:
            kwargs["d_model"] = self.d_model
        if "num_layers" in supported:
            kwargs["num_layers"] = self.num_layers
        if "discrete_mode" in supported:
            kwargs["discrete_mode"] = self.discrete_mode
        if "num_clusters" in supported:
            kwargs["num_clusters"] = self.num_clusters
        print(f"[ModelController:init] BiMamba ctor kwargs={kwargs}")
        try:
            return BiMambaMSM(**kwargs)
        except Exception as exc:
            if self.allow_fallback:
                print(f"[ModelController:init] fallback requested but disabled by default; build error={exc!r}")
            raise RuntimeError(f"Failed to construct BiMambaMSM with kwargs={kwargs}: {exc}") from exc

    @staticmethod
    def _has_prefix(keys: List[str], prefixes: Tuple[str, ...]) -> bool:
        return any(any(k.startswith(p) for p in prefixes) for k in keys)

    def _validate_checkpoint_compatibility(
        self,
        state: Dict[str, torch.Tensor],
        missing_keys: List[str],
        unexpected_keys: List[str],
    ) -> None:
        backbone_prefixes = ("proj_in.", "backbone.", "layer_norms.", "final_norm.", "input_norm.")
        if not self._has_prefix(list(state.keys()), backbone_prefixes):
            raise RuntimeError("Checkpoint lacks expected backbone keys.")
        missing_backbone = [k for k in missing_keys if k.startswith(backbone_prefixes)]
        unexpected_backbone = [k for k in unexpected_keys if k.startswith(backbone_prefixes)]
        if missing_backbone:
            raise RuntimeError(
                f"Backbone keys missing from checkpoint load: count={len(missing_backbone)} "
                f"sample={missing_backbone[:5]}"
            )
        too_many_unexpected = len(unexpected_keys) > max(32, len(state) // 4)
        if unexpected_backbone or too_many_unexpected:
            raise RuntimeError(
                f"Checkpoint compatibility mismatch: unexpected_backbone={len(unexpected_backbone)} "
                f"unexpected_total={len(unexpected_keys)}"
            )

    def single_inference(self, input_data: Any, sample_rate: int = 16000) -> Any:
        """Run inference on a single audio waveform and return frame-level reps [T, D].

        Args:
            input_data (torch.Tensor): A 1D tensor containing audio samples (shape [T])
            sample_rate (int): Sample rate of the provided audio.
        """
        if isinstance(input_data, dict):
            return self.single_evaluation(input_data)
        if not torch.is_tensor(input_data):
            raise TypeError("input_data must be a torch.Tensor waveform or a dict payload")

        if input_data.dim() == 2 and input_data.size(0) == 1:
            input_data = input_data.squeeze(0)
        if input_data.dim() != 1:
            raise ValueError(f"input_data must be 1D (shape [T]); got shape={tuple(input_data.shape)}")

        with torch.no_grad():
            wav = input_data.detach().to("cpu", dtype=torch.float32).unsqueeze(0)  # [1, T]
            wav = torch.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0)
            if sample_rate != self.SR:
                wav = torchaudio.functional.resample(wav, sample_rate, self.SR)
            x = self._wav_to_logmel_bt80(wav)
            reps = self._extract_backbone_reps(x)
            reps = torch.nan_to_num(reps, nan=0.0, posinf=0.0, neginf=0.0)
            self._maybe_print_inference_debug(
                wav_len=int(wav.shape[-1]),
                mel_shape=tuple(x.shape),
                hidden_shape=tuple(reps.shape),
            )
            return reps.squeeze(0).detach().cpu().float()

    def _decode_wav_b64(self, wav_b64: str) -> torch.Tensor:
        wav_bytes = base64.b64decode(wav_b64)
        try:
            wav, in_sr = torchaudio.load(io.BytesIO(wav_bytes))  # [C, N]
        except Exception:
            # Some torchaudio backends cannot decode file-like objects.
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                    f.write(wav_bytes)
                    f.flush()
                    wav, in_sr = torchaudio.load(f.name)
            except Exception:
                wav, in_sr = self._decode_wav_bytes_stdlib(wav_bytes)
        wav = torch.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if in_sr != self.SR:
            wav = torchaudio.functional.resample(wav, in_sr, self.SR)
        return wav  # [1, N]

    def _wav_to_logmel_bt80(self, wav: torch.Tensor) -> torch.Tensor:
        wav = torch.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0)
        wav = torch.clamp(wav, min=-1.0, max=1.0)
        if wav.size(-1) < self.N_FFT:
            wav = F.pad(wav, (0, self.N_FFT - wav.size(-1)))
        mel = self.mel_fn(wav)  # [1, 80, T]
        mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
        mel = torch.log(torch.clamp(mel, min=self.EPS))
        mel = torch.clamp(mel, min=-20.0, max=20.0)
        mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
        mel = mel.squeeze(0).transpose(0, 1).unsqueeze(0).contiguous()  # [1, T, 80]
        return mel.to(self.device, dtype=torch.float32)

    def _extract_backbone_reps(self, x_bt80: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Backbone model is not initialized.")
        with torch.no_grad():
            x_bt80 = x_bt80.to(device=next(self.model.parameters()).device, dtype=torch.float32).contiguous()
            forward_sig = inspect.signature(self.model.forward)
            call_kwargs: Dict[str, Any] = {}
            if "return_reps" in forward_sig.parameters:
                call_kwargs["return_reps"] = True
            out = self.model(x_bt80, **call_kwargs) if call_kwargs else self.model(x_bt80)
            hidden = self._coerce_hidden_from_forward_output(out)
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
            if hidden.ndim != 3:
                raise RuntimeError(f"Expected hidden ndim=3 [B,T,D], got shape={tuple(hidden.shape)}")
            if hidden.shape[0] != x_bt80.shape[0] or hidden.shape[1] != x_bt80.shape[1]:
                raise RuntimeError(
                    "Backbone hidden shape mismatch: "
                    f"expected [B,T,D] with B={x_bt80.shape[0]},T={x_bt80.shape[1]}, "
                    f"got shape={tuple(hidden.shape)}"
                )
            return hidden

    @staticmethod
    def _coerce_hidden_from_forward_output(out: Any) -> torch.Tensor:
        if torch.is_tensor(out):
            return out
        if isinstance(out, tuple):
            if len(out) >= 2 and torch.is_tensor(out[1]):
                return out[1]
            if len(out) >= 1 and torch.is_tensor(out[0]):
                return out[0]
            raise RuntimeError("Tuple output did not contain tensor hidden representations.")
        if isinstance(out, dict):
            for key in ("hidden", "reps", "last_hidden_state"):
                val = out.get(key)
                if torch.is_tensor(val):
                    return val
            raise RuntimeError(f"Dict output missing hidden representation keys: {list(out.keys())}")
        raise RuntimeError(f"Unsupported forward output type: {type(out).__name__}")

    def _masked_mean_pool(self, reps_btD: torch.Tensor, x_bt80: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        if reps_btD.ndim != 3 or x_bt80.ndim != 3:
            raise RuntimeError("Expected reps and mel inputs to both be rank-3 tensors.")
        if reps_btD.shape[:2] != x_bt80.shape[:2]:
            raise RuntimeError(
                f"Temporal shape mismatch between reps {tuple(reps_btD.shape)} and mel {tuple(x_bt80.shape)}"
            )
        # abs(log-mel) is wrong for silence masking because silence has large negative values.
        frame_energy = x_bt80.squeeze(0).mean(dim=-1)  # [T]
        frame_mask = frame_energy > self.VALID_FRAME_THRESHOLD
        if not frame_mask.any():
            frame_mask = torch.ones_like(frame_mask, dtype=torch.bool)
        pooled = reps_btD.squeeze(0)[frame_mask].mean(dim=0, keepdim=True)
        valid_frames = int(frame_mask.sum().item())
        total_frames = int(frame_mask.numel())
        return pooled, valid_frames, total_frames

    def _project_to_output_dim(self, emb: torch.Tensor) -> torch.Tensor:
        emb_dim = int(emb.numel())
        if emb_dim == self.OUTPUT_DIM:
            return emb.contiguous()
        if not self._dim_warning_printed:
            print(
                "[ModelController] WARNING: embedding dim mismatch "
                f"(got {emb_dim}, expected {self.OUTPUT_DIM}); applying pad/truncate."
            )
            self._dim_warning_printed = True
        if emb_dim > self.OUTPUT_DIM:
            return emb[: self.OUTPUT_DIM].contiguous()
        return F.pad(emb, (0, self.OUTPUT_DIM - emb_dim))

    def single_evaluation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if "wav_b64" not in payload:
            raise ValueError("payload must contain 'wav_b64'")
        wav = self._decode_wav_b64(payload["wav_b64"])
        x = self._wav_to_logmel_bt80(wav)
        reps = self._extract_backbone_reps(x)  # [1, T, D]
        pooled, valid_frames, total_frames = self._masked_mean_pool(reps, x)
        emb = pooled.squeeze(0)
        emb = self._project_to_output_dim(emb)
        emb = F.normalize(emb, dim=-1)  # L2 normalize
        self._maybe_print_inference_debug(
            wav_len=int(wav.shape[-1]),
            mel_shape=tuple(x.shape),
            hidden_shape=tuple(reps.shape),
            valid_frames=valid_frames,
            total_frames=total_frames,
            embedding_norm=float(emb.norm(p=2).item()),
        )
        return {"embedding": emb.detach().cpu().tolist()}

    def batch_evaluation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        items: List[Dict[str, Any]] = payload.get("items", [])
        return {"results": [self.single_evaluation(it) for it in items]}

    # Backwards/alternate API name used by some Dynabench runners
    def batch_inference(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.batch_evaluation(payload)

    def single_inference_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.single_evaluation(payload)

    def batch_inference_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.batch_evaluation(payload)

    def _maybe_print_inference_debug(
        self,
        wav_len: int,
        mel_shape: Tuple[int, ...],
        hidden_shape: Tuple[int, ...],
        valid_frames: Optional[int] = None,
        total_frames: Optional[int] = None,
        embedding_norm: Optional[float] = None,
    ) -> None:
        if self._inference_debug_printed:
            return
        msg = (
            "[ModelController:inference] "
            f"wav_len={wav_len} mel_shape={mel_shape} hidden_shape={hidden_shape}"
        )
        if valid_frames is not None and total_frames is not None:
            msg += f" valid_frames={valid_frames}/{total_frames}"
        if embedding_norm is not None:
            msg += f" embedding_norm={embedding_norm:.6f}"
        print(msg)
        self._inference_debug_printed = True

    @staticmethod
    def _self_test_example() -> None:
        """
        Tiny self-test usage example:
            m = Model()
            wav = torch.randn(16000)
            reps = m.single_inference(wav, sample_rate=16000)
            print("single_inference reps:", tuple(reps.shape))
            pcm16 = (wav.clamp(-1, 1).numpy() * 32767.0).astype(np.int16)
            bio = io.BytesIO()
            with wave.open(bio, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm16.tobytes())
            payload = {"wav_b64": base64.b64encode(bio.getvalue()).decode("ascii")}
            out = m.single_evaluation(payload)
            print("single_evaluation embedding dim:", len(out["embedding"]))
        """
        return None


Model = ModelController
