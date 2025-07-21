
in terminal 1 

cd catkin_ws

source devel/setup.bash

roslaunch my_gurdy_description one_touch_spawn_control.launch

in terminal 2 

pip install pyquaternion

cd catkin_ws

source devel/setup.bash

and then try out all these

roslaunch my_gurdy_description train_gurdy.launch

roslaunch my_gurdy_description launch_sarsa.launch

roslaunch my_gurdy_description launch_dqn.launch

roslaunch my_gurdy_description ppo_launch.launch

roslaunch my_gurdy_description sac_launch.launch

roslaunch my_gurdy_description ppo_launch.launch






