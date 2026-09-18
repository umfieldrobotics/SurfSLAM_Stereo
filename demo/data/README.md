All images in this directory were captured at shipwrecks managed by the National Oceanic and Atmospheric Administration Thunder Bay National Marine Sanctuary. A full catalog of the sanctuary's wrecks is available at [thunderbay.noaa.gov/shipwrecks](https://thunderbay.noaa.gov/shipwrecks/).

## Wrecks

| Directory | Wreck | Vessel type | Built | Lost | Depth |
| --- | --- | --- | --- | --- | --- |
| `bissel_wreckage` | [Harvey Bissell](https://thunderbay.noaa.gov/shipwrecks/harvey_bissell.html) | Wooden three-masted schooner barge | 1866 | 1905 | 15 ft |
| `monohansett_engine`, `monohansett_hull`, `monohansett_propellor` | [Monohansett](https://thunderbay.noaa.gov/shipwrecks/monohansett.html) | Wooden steam barge | 1872 | 1907 | 18 ft |
| `montana_engine` | [Montana](https://thunderbay.noaa.gov/shipwrecks/montana.html) | Wooden steam barge | 1872 | 1914 | 66 ft |
| `wilson_windlass` | [D.M. Wilson](https://thunderbay.noaa.gov/shipwrecks/d_m_wilson.html) | Wooden bulk freighter | 1873 | 1894 | 40 ft |

## These images are raw

Each pair is a **raw, distorted** ZED frame, straight off the camera and
untouched. Stereo matching needs a rectified pair, so rectify before running a
network on them -- `python -m demo --scene <name>` does it for you, using
`stereo_calib.yaml` in this directory.

`stereo_calib.yaml` is a copy of `SUDS_STEREO/data/calibration/stereo_calib.yaml`
from the [data release](https://deepblue.lib.umich.edu/data/concern/data_sets/r781wh411).
It is vendored here so the demo runs from a clone of this repository, without
downloading the dataset. Rectified, the rig is fx = 1692.5 px with a 0.1200 m
baseline, so depth in metres is `203.0 / disparity_in_pixels`. (The rectified
focal is larger than the raw 1412 px because `alpha=0` crops the undistorted
image to its largest all-valid rectangle.)

## Provenance

Every pair is a frame from the SUDS **test** split, so predictions on them can
be scored against the release's ground truth:

| Directory | SUDS scene | Frame |
| --- | --- | --- |
| `bissel_wreckage` | `bissel_candidate3` | `1749063480002936000` |
| `monohansett_engine` | `monohansett_candidate3` | `1749133052659280000` |
| `monohansett_hull` | `monohansett_candidate5` | `1748882348766336000` |
| `monohansett_propellor` | `monohansett_candidate4` | `1749133152609423000` |
| `montana_engine` | `montana_candidate1` | `1749144765651096000` |
| `wilson_windlass` | `wilson_candidate3` | `1749229903439309000` |
