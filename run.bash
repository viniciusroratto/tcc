source ./devel/setup.bash
gnome-terminal --tab -e "roslaunch tcc_gazebo tcc.launch" 
sleep 2
gnome-terminal --tab -e "rosrun tcc_pack talker.py"
gnome-terminal --tab -e "rosrun tcc_pack listener.py"
gnome-terminal --tab -e "rosrun tcc_pack move.py"
sleep 60
rosnode kill -a
pkill roslaunch
#rosservice call /gazebo/reset_world
