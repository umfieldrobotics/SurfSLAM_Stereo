"""Run a released SurfSLAM stereo model on a pair of images.

Two front ends over one inference core:

    python -m demo --scene monohansett_hull     # command line
    python -m demo.web                          # browser UI

Both call :func:`demo.core.predict_pair`. Nothing here is imported by training.
"""
