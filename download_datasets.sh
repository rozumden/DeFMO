## TODO: insert your dataset path
MYPATH=/cluster/home/denysr/scratch/dataset/

# falling objects dataset
mkdir ${MYPATH}falling_objects
wget http://ptak.felk.cvut.cz/personal/rozumden/falling-imgs.zip -P ${MYPATH}falling_objects
unzip ${MYPATH}falling_objects/falling-imgs.zip -d ${MYPATH}falling_objects/
rm ${MYPATH}falling_objects/falling-imgs.zip

# TbD-3D dataset
mkdir ${MYPATH}TbD-3D
wget http://cmp.felk.cvut.cz/fmo/files/TbD-3D-imgs.zip -P ${MYPATH}TbD-3D
unzip ${MYPATH}TbD-3D/TbD-3D-imgs.zip -d ${MYPATH}TbD-3D/
rm ${MYPATH}TbD-3D/TbD-3D-imgs.zip

# TbD dataset
mkdir ${MYPATH}TbD_GC
wget http://cmp.felk.cvut.cz/fmo/files/TbD.zip -P ${MYPATH}TbD_GC
unzip ${MYPATH}TbD_GC/TbD.zip -d ${MYPATH}TbD_GC/
rm ${MYPATH}TbD_GC/TbD.zip
