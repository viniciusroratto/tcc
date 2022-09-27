source ./devel/setup.bash
gnome-terminal --tab -e "roslaunch tcc_gazebo tcc.launch"
sleep 5
gnome-terminal --tab -e "python run_2b.py"


