## Simultaneous Localization and Interaction Force Estimation
Dependencies:
* [sparcsnode](https://github.com/nicola-lissandrini/sparcsnode)
* [libtorch](https://download.pytorch.org/libtorch/nightly/cu113/libtorch-cxx11-abi-shared-with-deps-latest.zip), installing: https://pytorch.org/cppdocs/installing.html

### Basic run
* `roslaunch slife slife.launch` launches main algorithm and wait for pointclouds/vicon to be published
* `rosbag play bagfiles/<any>` playback recorded pointclouds/vicon
All configuration parameters in `config/slife.yaml`
