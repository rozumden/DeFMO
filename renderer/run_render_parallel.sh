step_size=2
for i in `seq 0 $step_size 50`; 
do 
python3 run_render.py $i $step_size &
done
