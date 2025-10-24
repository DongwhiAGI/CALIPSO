# CALIPSO: Compound Array of Lensless In-sensor Processing for Seeking Objects
Code and data for a **Nature Electronics** submission.
This project uses sensor-captured measurements and trains models that predict the 3D position of a light source from those data, with tools for visualization.

---

## Prepared Features

### 1) Train

Train on the provided simplified demo dataset.
Training on the full dataset took **9 h 13 m** , and the resulting checkpoint is saved at:

```
outputs/checkpoints/square_best_model.pth and
outputs/checkpoints/spiral_best_model.pth
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

* **square** (2 s)

  * **Via helper script**

    ```bash
    scripts/run_infer --config configs/demo/infer_square_demo.yaml --mode batch
    ```
  * **Direct Python invocation**

    ```bash
    python -m src.infer --config configs\demo\infer_square_demo.yaml --mode batch
    ```

* **spiral** (6 s)

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

* **After inference, render and save a final plot**(square: 25 s, spiral: 1 m 40 s)

  * **Via helper script**

    ```bash
    scripts/run_make_plot --config configs/demo/square_visualize_demo.yaml --style final --draw-every 1 --out outputs/square_plotly.html
    ```
  * **Direct Python invocation**

    ```bash
    python -m src.make_plot --config configs\demo\square_visualize_demo.yaml --style final --draw-every 1 --out outputs\square_plotly.html
    ```

* **Incremental plotting while inference progresses**(If the Plotly figure does not appear in your environment, run this in a Jupyter environment instead.)

  * **Via helper script**

    ```bash
    scripts/run_make_plot --config configs/demo/square_visualize_demo.yaml --style incremental --draw-every 1
    ```
  * **Direct Python invocation**

    ```bash
    python -m src.make_plot --config configs\demo\square_visualize_demo.yaml --style incremental --draw-every 1
    ```

* **Incremental plotting with interactive OpenGL**(square: 35 s, spiral: 2 m 20 s)

  * **Via helper script**

    ```bash
    scripts/run_make_plot --config configs/demo/square_visualize_demo.yaml --style incremental_opengl --draw-every 1
    ```
  * **Direct Python invocation**

    ```bash
    python -m src.make_plot --config configs\demo\square_visualize_demo.yaml --style incremental_opengl --draw-every 1
    ```

---

### 4) Video

Generate a video from the visualization.
As inference runs, results are rendered via OpenGL; after completion, they are encoded into a video file.(square: 1 m 35 s, spiral: 6 m 10 s)


* **Via helper script**

  ```bash
  scripts/run_make_video --config configs/demo/square_visualize_demo.yaml --fps 240 --draw-every 1 --out outputs/square_opengl.mp4
  ```
* **Direct Python invocation**

  ```bash
  python -m src.make_video --config configs\demo\square_visualize_demo.yaml --fps 240 --draw-every 1 --out outputs\square_opengl.mp4
  ```

---

### 5) Model Info

Print model details.

* **Via helper script**

  ```bash
  scripts/run_model_info --config configs/demo/infer_square_demo.yaml
  ```
* **Direct Python invocation**

  ```bash
  python -m src.tools.print_model_info --config configs\demo\infer_square_demo.yaml
  ```
