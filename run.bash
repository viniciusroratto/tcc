source ./devel/setup.bash
gnome-terminal --tab -e "roslaunch tcc_gazebo tcc.launch" 
sleep 5
#gnome-terminal --tab -e "rosrun tcc_pack talker.py"
#gnome-terminal --tab -e "rosrun tcc_pack listener.py"
gnome-terminal --tab -e "rosrun tcc_pack move.py"
sleep 1000
rosnode kill -a
pkill roslaunch
convert -delay 40 -loop 0 ./results/*.jpg sequence.gif
#rosservice call /gazebo/reset_world
#rosservice call /gazebo/get_world_properties "{}"

