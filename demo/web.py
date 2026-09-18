"""Browser demo: drag in a stereo pair, or click one of the shipwrecks.

    python -m demo.web

Serves on http://localhost:7860. ``docker/run.sh`` runs with ``--network=host``,
so the port is reachable from the host without any mapping.

This is a front end and nothing else -- every prediction goes through
:func:`demo.core.predict_pair`, the same call the CLI makes.
"""

from __future__ import annotations

import argparse
import tempfile

import gradio as gr

from demo import core

TITLE = "SurfSLAM: Sim-to-Real Underwater Stereo Reconstruction For Real-Time SLAM"

DESCRIPTION = f"""
# {TITLE}

Estimate disparity and metric depth from an underwater stereo pair.

<div id="badges">
<a href="https://arxiv.org/abs/1234.56789"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2601.10814-b31b1b.svg"></a>
<a href="https://umfieldrobotics.github.io/SurfSLAM/"><img alt="Project Page" src="https://img.shields.io/badge/Project_Page-SurfSLAM-blue"></a>
<a href="https://deepblue.lib.umich.edu/data/concern/data_sets/r781wh411"><img alt="DeepBlue" src="https://img.shields.io/badge/DeepBlue-Dataset_and_Weights-00274C"></a>
</div>
"""

EXAMPLES_INTRO = """
### Example Data
Raw frames from shipwreck surveys at the
[NOAA Thunder Bay National Marine Sanctuary](https://thunderbay.noaa.gov/shipwrecks/).
Click one to load it, or scroll down and drop in your own pair.
"""

# The examples panel is the first thing to try, so it gets a box of its own at
# the top. The buttons themselves stay compact: gradio pads every example to a
# fixed 200px of text width, which leaves a row of mostly-empty boxes.
CSS = """
/* gradio's prose styling makes markdown images block-level, which stacks the
   shield badges one per line. */
#badges { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
#badges img { display: inline-block; margin: 0; height: 20px; }

#examples-panel {
    border: 2px solid var(--color-accent);
    border-radius: 10px;
    padding: 4px 16px 14px 16px;
    margin-bottom: 18px;
    background: var(--background-fill-secondary);
}
#examples-panel .label { display: none; }   /* an empty label row with a stray icon */
#examples-panel .gallery { gap: 8px; }
#examples-panel button.gallery-item {
    font-size: 0.85rem !important;
    padding: 5px 10px !important;
    border: 1px solid var(--color-accent) !important;
}
#examples-panel button.gallery-item > div {
    --local-text-width: auto !important;
    width: auto !important;
}
#examples-panel button.gallery-item:hover {
    background: var(--color-accent) !important;
    color: white !important;
}
"""

FONT = ["system-ui", "-apple-system", "Segoe UI", "Roboto",
        "Helvetica Neue", "Arial", "sans-serif"]
FONT_MONO = ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"]

NO_WEIGHTS = """
### No checkpoints found

Download the released weights and point `model_weights` in `config/dataset_paths.yaml`
at them, then restart. `python -m demo --list-checkpoints` shows what resolves.
"""


def run_inference(left, right, rectify, checkpoint_label, checkpoint_path, architecture,
                  scale, iters, calib_file, focal_px, baseline_m, max_depth,
                  min_disparity, device):
    """Gradio callback: returns (disparity, depth, panel, npz path, status)."""
    if left is None or right is None:
        raise gr.Error("Two images please: a left and a right view of the same scene.")

    checkpoint = (checkpoint_path or "").strip() or checkpoint_label
    if not checkpoint:
        raise gr.Error("Pick a checkpoint.")

    rectification = None
    if rectify:
        calib = calib_file or core.DEMO_CALIB_FILE
        try:
            rectification = core.load_kalibr_calibration(str(calib))
        except Exception as err:
            raise gr.Error(f"Could not read that calibration: {err}")

    try:
        result = core.predict_pair(
            left, right,
            checkpoint=checkpoint,
            model=architecture,
            rectification=rectification,
            focal_px=focal_px or None,
            baseline_m=baseline_m or None,
            scale=scale,
            iters=int(iters),
            min_disparity=min_disparity if min_disparity is not None else core.MIN_DISPARITY_PX,
            device=device,
        )
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as err:
        raise gr.Error(str(err))

    written = core.write_outputs(result, tempfile.mkdtemp(prefix="surfslam_demo_"),
                                 max_depth=max_depth or None)

    status = (f"**{result.checkpoint}** &middot; {result.width}&times;{result.height} &middot; "
              f"{result.iters} iters &middot; {result.runtime_s:.1f} s\n\n```\n"
              f"{result.summary()}\n```")

    return (result.disparity_image(),
            result.depth_image(max_depth or None),
            core.make_panel(result),
            str(written["arrays"]),
            status)


def build_ui(device: str = "cuda") -> gr.Blocks:
    checkpoints = core.list_checkpoints()
    labels = [c.label for c in checkpoints]
    default = core.DEFAULT_CHECKPOINT if core.DEFAULT_CHECKPOINT in labels else (
        labels[0] if labels else None)
    scenes = core.list_scenes()

    theme = gr.themes.Soft(font=FONT, font_mono=FONT_MONO)
    with gr.Blocks(title=TITLE, theme=theme, css=CSS) as ui:
        gr.Markdown(DESCRIPTION)
        if not labels:
            gr.Markdown(NO_WEIGHTS)

        # Built first but rendered below, so the examples can sit at the top
        # where people will actually see them.
        left = gr.Image(type="numpy", image_mode="RGB", label="Left", render=False)
        right = gr.Image(type="numpy", image_mode="RGB", label="Right", render=False)
        rectify = gr.Checkbox(
            value=False, render=False,
            label="Rectify with our calibration for SUDS data",
            info="Required for examples. If using your own data, supply calibration in `advanced'.")
        checkpoint = gr.Dropdown(choices=labels, value=default, label="Checkpoint",
                                 info="Released weights found on this machine.",
                                 render=False)

        if scenes:
            with gr.Column(elem_id="examples-panel"):
                gr.Markdown(EXAMPLES_INTRO)
                gr.Examples(
                    examples=[[str(s.left), str(s.right), True, default] for s in scenes],
                    inputs=[left, right, rectify, checkpoint],
                    # Named buttons rather than a thumbnail table: the table repeats
                    # the rectify and checkpoint columns on every row, which is a lot
                    # of page for no information.
                    example_labels=[s.name.replace("_", " ") for s in scenes],
                    label="",
                    cache_examples=False,
                )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    left.render()
                    right.render()

                rectify.render()
                checkpoint.render()

                with gr.Accordion("Advanced", open=False):
                    checkpoint_path = gr.Textbox(
                        label="Checkpoint path",
                        placeholder="/path/to/best.pth  (overrides the dropdown)")
                    architecture = gr.Dropdown(
                        choices=core.list_models(), value=core.DEFAULT_MODEL,
                        label="Architecture for that path")
                    scale = gr.Slider(0.25, 1.0, value=core.DEFAULT_SCALE, step=0.05,
                                      label="Inference scale",
                                      info="Fraction of full resolution the network runs at. "
                                           "Lower is faster and blurrier.")
                    iters = gr.Slider(4, 64, value=32, step=1, label="Refinement iterations")
                    calib_file = gr.File(label="Calibration (Kalibr .yaml)",
                                         file_types=[".yaml", ".yml"], type="filepath")
                    with gr.Row():
                        focal_px = gr.Number(label="Focal length (px)", value=None)
                        baseline_m = gr.Number(label="Baseline (m)", value=None)
                    max_depth = gr.Number(label="Clip depth colormap at (m)", value=None)
                    min_disparity = gr.Number(
                        label="Minimum disparity (px)", value=core.MIN_DISPARITY_PX,
                        info="Below this, a pixel is open water rather than a "
                             "distance, and gets no depth.")

                run = gr.Button("Estimate depth", variant="primary",
                                interactive=bool(labels))

            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("Disparity"):
                        disparity_out = gr.Image(label="Disparity (px)")
                    with gr.Tab("Depth"):
                        depth_out = gr.Image(label="Depth (m)")
                    with gr.Tab("Panel"):
                        panel_out = gr.Image(label="Left / disparity / depth")
                status = gr.Markdown()
                arrays_out = gr.File(label="Raw arrays (.npz)")

        device_state = gr.State(device)
        run.click(
            fn=run_inference,
            inputs=[left, right, rectify, checkpoint, checkpoint_path, architecture,
                    scale, iters, calib_file, focal_px, baseline_m, max_depth,
                    min_disparity, device_state],
            outputs=[disparity_out, depth_out, panel_out, arrays_out, status],
        )

    return ui


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m demo.web",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0", help="interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="port (default: 7860)")
    parser.add_argument("--share", action="store_true",
                        help="also expose a temporary public gradio.live URL")
    parser.add_argument("--device", default="cuda", help="torch device (default: cuda)")
    parser.add_argument("--preload", action="store_true",
                        help="load the default checkpoint at startup so the first "
                             "prediction is not a cold start")
    args = parser.parse_args(argv)

    if args.preload:
        try:
            ckpt = core.find_checkpoint(core.DEFAULT_CHECKPOINT)
            print(f"preloading {ckpt.label} ...", flush=True)
            core.get_predictor(ckpt.model, ckpt.path, args.device)
        except (FileNotFoundError, KeyError, RuntimeError) as err:
            print(f"warning: could not preload a model ({err})")

    ui = build_ui(device=args.device)
    ui.queue(default_concurrency_limit=1)   # one GPU job at a time
    ui.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
