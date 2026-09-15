# Anima Autograd & Training Validation Log

本ドキュメントは、Anima Slider LoRA における autograd 互換性、gradient checkpointing、および診断ログの検証記録を管理するものです。

## 1. 検証環境と基準

- **Baseline Commit**: `aeb1688226cf1d93f91ddfdb9c80742056205418`
- **Diagnostic Branch**: `diagnostic/anima-autograd-rms-rope`
- **Local Test Environment**: Windows 11 / Python 3.10.11 / PyTorch 2.10.0 (CPU) / pytest 9.0.2

## 2. 検証記録マトリクス

| 試験ID | 場所 | 状態 | commit | Python/torch/ck/ComfyUI | depth/settings/seed | 最終phase・観測 | 証拠 |
|---|---|---|---|---|---|---|---|
| T-CPU-01 | ローカルCPU | PASS | e0ca2fe | 3.10 / 2.10.0 / fake / なし | synthetic custom op / seed=123 | `autograd_safe_comfy_kitchen_rope` 下で未登録autograd例外が解消、loss.backward()成功、grad有限性確認 | `tests/test_autograd_compat.py::test_rms_backward_under_training_context` |
| T-CPU-02 | ローカルCPU | PASS | e0ca2fe | 3.10 / 2.10.0 / fake / なし | context normal/exception/nested | context退出時および例外発生時に元のバインディングが100%復元されることを確認 | `tests/test_autograd_compat.py` (restoration & nested) |
| T-CPU-03 | ローカルCPU | PASS | e0ca2fe | 3.10 / 2.10.0 / fake / なし | TinyBlock + TinyRootModel / checkpoint=True/False | checkpoint on/offで出力・loss・勾配が一致、checkpoint時のブロック呼出回数が2回に増加、base weight不変、LoRAパラメータのAdamW更新確認 | `tests/test_training_autograd_integration.py` |
| T-CPU-04 | ローカルCPU | PASS | 診断版HEAD | 3.10 / 2.10.0 / fake / なし | debug=1 / debug_session | `[AnimaSliderDebug]` プレフィックス付きJSONログ出力、同一run_id維持、schema_version=1、backward_end/save_end確認 | `tests/test_training_debug.py::test_one_session_keeps_run_id_and_emits_valid_json` |
| T-CPU-05 | ローカルCPU | PASS | 診断版HEAD | 3.10 / 2.10.0 / fake / なし | debug=0 vs debug=1 / seed=42 | debugの有無で順伝播出力・loss値・LoRA勾配・更新後重みがビット単位で完全一致（非干渉確認） | `tests/test_training_debug.py::test_debug_non_interference_outputs_and_gradients` |
| T-CPU-06 | ローカルCPU | PASS | 診断版HEAD | 3.10 / 2.10.0 / fake / なし | debug=0 / mock snapshot | debug=0 時に snapshot や不要なフック処理が実行されないことを確認 | `tests/test_training_debug.py::test_debug_off_does_not_call_snapshot_or_cuda_sync` |
| T-CPU-07 | ローカルCPU | PASS | 診断版HEAD | 3.10 / 2.10.0 / fake / なし | exception simulation | phase内で発生した例外が `run_error` として記録され、元の例外型・メッセージ・tracebackが完全に保持されることを確認 | `tests/test_training_debug.py::test_run_error_and_traceback_preservation` |
| A1 | クラウドGPU | NOT_RUN | 未定 | 3.14.5 / 2.13.0+cu130 / ck 0.2.33 / ComfyUI v0.35.0-33 | 40 blocks, 512x512, 1 step, prompt0, eval skip | クラウド確認A待ち | 利用者実行待ち |
| A2 | クラウドGPU | NOT_RUN | 未定 | 3.14.5 / 2.13.0+cu130 / ck 0.2.33 / ComfyUI v0.35.0-33 | 512x512, 3 steps, checkpoint=True | クラウド確認A待ち | 利用者実行待ち |
| A3 | クラウドGPU | NOT_RUN | 未定 | 3.14.5 / 2.13.0+cu130 / ck 0.2.33 / ComfyUI v0.35.0-33 | 1024x1024, 1 step, eval skip | クラウド確認A待ち | 利用者実行待ち |

## 3. 未確認事項（ローカルCPU環境では判定不可）

- 実CUDA環境での `comfy_kitchen.rms_rope_split_half` カーネル実行と backward
- 1024x1024 における実VRAM消費量および OOM の有無
- 28ブロック版 Anima での実機動作
- QPOLA オプティマイザの実機動作
- 900 steps 等の長時間学習における画質・収束性
