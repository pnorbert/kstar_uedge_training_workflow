# source me
pyenv-adios
pathmunge /opt/adios2/bin
pymunge /opt/adios2/home/adios/.virtualenvs/adios/lib/python3.12/site-packages
#
#pymunge /home/adios/shared/Software/hpc-campaign/source
export PYTHONPATH
export PATH

# Add nvidia libraries installed with tensorflow manually
export LD_LIBRARY_PATH=$(find "$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia" -type d -name lib | paste -sd:):$LD_LIBRARY_PATH
