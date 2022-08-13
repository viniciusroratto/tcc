#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/vini/Desktop/tcc/src/gazebo_ros_pkgs/gazebo_ros"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/vini/Desktop/tcc/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/vini/Desktop/tcc/install/lib/python3/dist-packages:/home/vini/Desktop/tcc/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/vini/Desktop/tcc/build" \
    "/home/vini/anaconda3/bin/python3" \
    "/home/vini/Desktop/tcc/src/gazebo_ros_pkgs/gazebo_ros/setup.py" \
     \
    build --build-base "/home/vini/Desktop/tcc/build/gazebo_ros_pkgs/gazebo_ros" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/vini/Desktop/tcc/install" --install-scripts="/home/vini/Desktop/tcc/install/bin"
