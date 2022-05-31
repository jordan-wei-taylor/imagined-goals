active_python=`which python3`
system_python="/usr/bin/python3"

if [[ $active_python == $system_python ]]; then
    echo "activate a virtual python environment first!"
else
    pip uninstall rlig -y; pip install .
fi
