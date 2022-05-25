# install dependencies
sudo apt install libosmesa6-dev libglew-dev patchelf

# copy mujoco directory
rm -rf ~/.mujoco
cp -r mujoco ~/.mujoco

# check if export line in .bashrc
line="export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/jordan/.mujoco/mujoco210/bin"
check=true
while read row; 
    do
        if [[ $row==$line ]]; then
            check=false
        fi
done < ~/.bashrc

# if export line missing add it to bashrc and load it
if [[ $check ]]; then

    echo $line >> ~/.bashrc
    source ~/.bashrc

fi

# if virtual python environment missing, create it
if [ ! -d env ]; then
    python3 -m venv env
fi

# activate virtual python environment, install mujoco-py and run an import command to run first time setup
source env/bin/activate
pip install mujoco-py
python3 -c "import mujoco_py"
