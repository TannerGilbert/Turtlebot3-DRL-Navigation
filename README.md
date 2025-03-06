# Turtlebot3 DRL Navigation

> [!WARNING]
> WIP convertion to Gazebo Fortress and Harmonic. Resetting of the environment and model position update doesn't work as expected yet:
> StackExchange issue: [ROS 2 call /world/default/set_pose Gazebo service](https://robotics.stackexchange.com/questions/114861/ros-2-call-world-default-set-pose-gazebo-service)
> As a workaround I have been using [Gazebo Transport](https://gazebosim.org/api/transport/13/python.html)
> If you get an error similar to:
>```
> [test_td3-9] Traceback (most recent call last):
> [test_td3-9]   File "/home/gilbert/Programming/Turtlebot3-DRL-Navigation/install/td3_rl/lib/td3_rl/test_td3", line 33, in <module>
> [test_td3-9]     sys.exit(load_entry_point('td3-rl', 'console_scripts', 'test_td3')())
> [test_td3-9]   File "/home/gilbert/Programming/Turtlebot3-DRL-Navigation/install/td3_rl/lib/td3_rl/test_td3", line 25, in importlib_load_entry_point
> [test_td3-9]     return next(matches).load()
> [test_td3-9]   File "/usr/lib/python3.10/importlib/metadata/__init__.py", line 171, in load
> [test_td3-9]     module = import_module(match.group('module'))
> [test_td3-9]   File "/usr/lib/python3.10/importlib/__init__.py", line 126, in import_module
> [test_td3-9]     return _bootstrap._gcd_import(name[level:], package, level)
> [test_td3-9]   File "<frozen importlib._bootstrap>", line 1050, in _gcd_import
> [test_td3-9]   File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
> [test_td3-9]   File "<frozen importlib._bootstrap>", line 1006, in _find_and_load_unlocked
> [test_td3-9]   File "<frozen importlib._bootstrap>", line 688, in _load_unlocked
> [test_td3-9]   File "<frozen importlib._bootstrap_external>", line 883, in exec_module
> [test_td3-9]   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
> [test_td3-9]   File "/home/gilbert/Programming/Turtlebot3-DRL-Navigation/build/td3_rl/td3_rl/test_td3.py", line 13, in <module>
> [test_td3-9]     from td3_rl.gazebo_env import GazeboEnv
> [test_td3-9]   File "/home/gilbert/Programming/Turtlebot3-DRL-Navigation/build/td3_rl/td3_rl/gazebo_env.py", line 9, in <module>
> [test_td3-9]     from gz.msgs10.boolean_pb2 import Boolean
> [test_td3-9]   File "/usr/lib/python3/dist-packages/gz/msgs10/boolean_pb2.py", line 14, in <module>
> [test_td3-9]     from gz.msgs import header_pb2 as gz_dot_msgs_dot_header__pb2
> [test_td3-9]   File "/usr/lib/python3/dist-packages/gz/msgs10/header_pb2.py", line 14, in <module>
> [test_td3-9]     from gz.msgs import time_pb2 as gz_dot_msgs_dot_time__pb2
> [test_td3-9]   File "/usr/lib/python3/dist-packages/gz/msgs10/time_pb2.py", line 36, in <module>
> [test_td3-9]     _descriptor.FieldDescriptor(
> [test_td3-9]   File "/home/gilbert/.local/lib/python3.10/site-packages/google/protobuf/descriptor.py", line 621, in __new__
> [test_td3-9]     _message.Message._CheckCalledFromGeneratedFile()
> [test_td3-9] TypeError: Descriptors cannot be created directly.
> [test_td3-9] If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
> [test_td3-9] If you cannot immediately regenerate your protos, some other possible workarounds are:
> [test_td3-9]  1. Downgrade the protobuf package to 3.20.x or lower.
> [test_td3-9]  2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
> [test_td3-9] 
> [test_td3-9] More information: https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates
```
> You can set `export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` as suggested in the error message.

This repository is a fork of [DRL-robot-navigation](https://github.com/reiniscimurs/DRL-robot-navigation) with modifications to run with the Turtlebot3 and its 2D lidar sensor. All the base behavior should be credited to the original author. The fork however changes the structure of the project so everything can be executed directly via ROS.

![Inference example](doc/model_test.gif)

## Installation

1. Clone the repository
    ```bash
    git clone -b ros2 https://github.com/TannerGilbert/DRL-robot-navigation --recursive
    ```
2. Install Python dependencies
    ```bash
    pip3 install -r requirements.txt
    ```
3. Build workspace
    ```bash
    colcon build --symlink-install
    ```

## Usage

### Train

Training can be started via:

```bash
ros2 launch td3_rl train_td3.launch.py
```

> Note: Training can be sped up by increasing the simulation speed in the Gazebo environment. This can be done by changing the `real_time_update_rate` in [`TD3.world`](src/TD3/worlds/TD3.world). For example setting it to `0` will make the simulation run as fast as possible.

![Gazebo environment](doc/environment.png)

![RVIZ](doc/rviz.png)

The training can be configured inside the [td3_config.yaml](src/TD3/config/td3_config.yaml) file.

Check training progress with Tensorboard:
```bash
$ cd src/TD3
$ tensorboard --logdir runs
```

![Tensorboard](doc/tensorboard.png)

### Inference

After training, the model can be tested with the following command:

```bash
ros2 launch td3_rl test_td3.launch.py
```

The default behaviour for the inference script is to run indefinitetly until the robot crashes. If the environment should also be reset after a certain time `td3_params/max_ep` needs to be set to a positive value.

## Acknowledgements

This repository builds on [the work](https://github.com/reiniscimurs/DRL-robot-navigation) of [Reinis Cimurs](https://github.com/reiniscimurs).