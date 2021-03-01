# BASE_PATH=/mnt/lascar/rozumden/dataset
BASE_PATH=/cluster/scratch/denysr/dataset

PATTERNS_PATH="${BASE_PATH}/patterns"
BG_PATH="${BASE_PATH}/vot"

mkdir -p  ${PATTERNS_PATH} 

VOT_YEAR=2016 bash trackdat/scripts/download_vot.sh dl/vot
bash trackdat/scripts/unpack_vot.sh dl/vot ${BG_PATH}
python3 generate_patterns.py --fg_path ${PATTERNS_PATH} 


