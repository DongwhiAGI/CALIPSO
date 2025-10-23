# CALIPSO: Compound Array of Lensless In-sensor Processing for Seeking Objects
Code and data for a **Nature Electronics** submission.
This project uses sensor-captured measurements and trains models that predict the 3D position of a light source from those data, with tools for visualization.

---

## Prepared Features

### 1) Train

Train on the provided simplified demo dataset.
Training on the full dataset took **00 h 00 m** (placeholder), and the resulting checkpoint is saved at:

```
outputs/checkpoints/best_model.pth
```

This checkpoint is used for inference.

**Via helper script**

```bash
scripts/run_train --config configs/demo/train_demo.yaml
```

**Direct Python invocation**

```bash
python -m src.train --config configs\demo\train_demo.yaml
```

---

### 2) Inference

Load the pretrained weights (from the full dataset) and run inference on the prepared evaluation data.
Results are saved as `.csv`.

Per-dataset runtimes and commands:

* **square** (0 min 0 s)

  * **Via helper script**

    ```bash
    scripts/run_infer --config configs/demo/infer_square_demo.yaml --mode batch
    ```
  * **Direct Python invocation**

    ```bash
    python -m src.infer --config configs\demo\infer_square_demo.yaml --mode batch
    ```

* **spiral** (0 min 0 s)

  * **Via helper script**

    ```bash
    scripts/run_infer --config configs/demo/infer_spiral_demo.yaml --mode batch
    ```
  * **Direct Python invocation**

    ```bash
    python -m src.infer --config configs\demo\infer_spiral_demo.yaml --mode batch
    ```

---

### 3) Plot

Visualize predictions.

* **After inference, render and save a final plot**

  * **Via helper script**

    ```bash
    scripts/run_make_plot --config configs/demo/square_visualize.yaml --style final --draw-every 1 --out outputs/square_plotly.html
    ```
  * **Direct Python invocation**

    ```bash
    python -m src.make_plot --config configs\demo\square_visualize.yaml --style final --draw-every 1 --out outputs\square_plotly.html
    ```

* **Incremental plotting while inference progresses**

  * **Via helper script**

    ```bash
    scripts/run_make_plot --config configs/demo/square_visualize.yaml --style incremental --draw-every 1
    ```
  * **Direct Python invocation**

    ```bash
    python -m src.make_plot --config configs\demo\square_visualize.yaml --style incremental --draw-every 1
    ```

* **Incremental plotting with interactive OpenGL**

  * **Via helper script**

    ```bash
    scripts/run_make_plot --config configs/demo/square_visualize.yaml --style incremental_opengl --draw-every 1
    ```
  * **Direct Python invocation**

    ```bash
    python -m src.make_plot --config configs\demo\square_visualize.yaml --style incremental_opengl --draw-every 1
    ```

---

### 4) Video

Generate a video from the visualization.
As inference runs, results are rendered via OpenGL; after completion, they are encoded into a video file.

* **Via helper script**

  ```bash
  scripts/run_make_video --config configs/demo/square_visualize.yaml --fps 240 --draw-every 1 --out outputs/square_opengl.mp4
  ```
* **Direct Python invocation**

  ```bash
  python -m src.make_video --config configs\demo\square_visualize.yaml --fps 240 --draw-every 1 --out outputs\square_opengl.mp4
  ```

---

### 5) Model Info

Print model details.

* **Via helper script**

  ```bash
  scripts/run_model_info --config configs/demo/infer_demo.yaml
  ```
* **Direct Python invocation**

  ```bash
  python -m src.tools.print_model_info --config configs\demo\infer_demo.yaml
  ```

---

> **Notes**
>
> * Replace placeholder times (e.g., **00 h 00 m**) with actual measurements once available.
> * Paths in examples show both Unix-style and Windows-style separators where relevant; use whichever matches your environment.