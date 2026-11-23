# Deep Learning – Assignment 02: CNN Architectures, Training & Ablation Studies

Air University · TensorFlow / Keras · Google Colab (NVIDIA T4) · CIFAR-10

## Folder contents
| File | Purpose |
|---|---|
| `cnn_assignment.ipynb` | the single notebook: Part I checks, data protocol, four architectures, LR sanity check, ablations A0–A4, analysis, final test, report generation |
| `src/models.py` | AlexNet, VGG16, GoogLeNet, ResNet-18, A0–A3 and A4 model code (written by the notebook's `%%writefile` cell, also included here) |
| `src/build_report.py` | writes `report.docx` (also written by the notebook) |
| `report.docx` | research-style Word report (Appendix A = Part I answers). Export to `report.pdf` from Word for submission |
| `results.csv`, `outputs/results.json` | machine-readable metrics of every run (created by the notebook) |
| `outputs/` | figures, histories, logs, `data_indices.json`, `environment.json`, `nvidia_smi.txt`, test record |
| `requirements.txt` | dependencies |

## Exact reproduction steps
1. Open `cnn_assignment.ipynb` in Google Colab (`File → Upload notebook`, or open it from GitHub).
2. `Runtime → Change runtime type → T4 GPU`.
3. *(optional, ~5 min)* set `SMOKE_TEST = True` in the config cell and run all, to check the pipeline on a tiny subset (results go to `outputs_smoke/`, `results_smoke.csv`). Set it back to `False`.
4. `Runtime → Restart session and run all` (clean start). The notebook downloads CIFAR-10 itself, uses **no Google-Drive paths**, and finishes by building `report.docx` and a download zip.
5. If Colab disconnects, run all again: finished runs found in `outputs/results.json` are skipped (set `FORCE_RERUN = True` to redo everything). Note that Colab's local disk is wiped when the runtime is recycled – download `cnn_assignment_submission.zip` once the run is done.
6. Open `report.docx`, read the auto-written interpretation paragraphs (highlighted in brown where you are asked to add your own), fill in your name / roll number, and export a PDF (`File → Save as → PDF`) as `report.pdf`.
7. Push everything to GitHub and put the output of `git rev-parse HEAD` into the commit-hash row of Appendix B (the notebook records it automatically only if it runs inside a git checkout).

## Protocol summary (as required by the assignment)
* CIFAR-10 only. Fixed stratified 12 000-image train subset + disjoint 3 000-image validation subset from the **original training partition** (seed 42, indices in `outputs/data_indices.json`). The 10 000-image test split is evaluated **once**, for a winner recorded in `outputs/final_model_selection.json` beforehand.
* Four architectures: 96×96 (bilinear resize inside `tf.data`), 5 epochs, one seed (42), no augmentation. Ablation A0–A3-P/R: 64×64, seeds 42 and 123, 10 epochs. A4-F/A4-T: separate transfer experiment (ImageNet ResNet50, 96×96).
* Adam 3e-4 (after a brief LR sanity check), `SparseCategoricalCrossentropy(from_logits=True)`, `mixed_float16` with float32 logits, batch 32 (16 on OOM, attempts logged), checkpoint = best validation accuracy.
* `keras.applications.VGG16(weights=None)` is used **only** as a structural cross-check; `keras.applications.ResNet50` **only** for A4. No PyTorch/torchvision.

## Honesty notes
* The report contains only numbers that the notebook measured. Failed / OOM runs stay in `outputs/logs/run_log.jsonl` and in the report.
* Single-seed results are provisional. You must be able to explain every architecture, experimental decision and line of code in the oral verification.
