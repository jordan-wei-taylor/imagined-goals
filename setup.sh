# install dependencies
sudo apt-get install libosmesa6-dev libglew-dev patchelf >/dev/null

echo "package dependencies are installed"

# copy mujoco directory
rm -rf ~/.mujoco
cp -r mujoco ~/.mujoco

echo "updated ~/.mujoco directory"

# check if export line in .bashrc
line="export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/$USER/.mujoco/mujoco210/bin"
check=true
while read row; 
    do
        if [[ $row==$line ]]; then
            check=false
        fi
done < ~/.bashrc

# if export line missing add it to bashrc and load it
if $check; then
    echo "adding \"$line\"" to ~/.bashrc file
    echo $line >> ~/.bashrc
    source ~/.bashrc

fi

# if virtual python environment missing, create it
if [ ! -d rlig-env ]; then
    echo "creating virtual python environment \"rlig-env\" with mujoco_py"
    python3 -m venv rlig-env
fi

# activate virtual python environment, install mujoco-py and run an import command to run first time setup
source rlig-env/bin/activate
pip install mujoco-py >/dev/null

if python3 -c "import mujoco_py" >/dev/null ; then
    echo "successfully installed mujoco_py"
else
    echo "something went wrong!"
fi

# install requirements.txt
pip install -r requirements.txt >/dev/null

# attempts to install mujoco extra option within gym (it fails but still works...)
pip install gym[mujoco] > /dev/null 2> /dev/null

# try making HalfCheetah
if python3 -c "import gym, warnings; warnings.filterwarnings('ignore'); gym.make('HalfCheetah')" >/dev/null; then
    # remove verbose prints from import distutils
    echo -en "\033[2K\033[1A\033[2K\033[1A\033[2K"
    echo "\"HalfCheetah\" gym environment build pass: true"
else
    echo "\"HalfCheetah\" gym environment build pass: false"
fi
