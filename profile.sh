git clone https://github.com/baler-collaboration/baler.git

#cProfile
python3 -m cProfile -o cProfile_train.prof  baler/baler.py --mode train --project profile cpu
python3 -m cProfile -o cProfile_compress.prof  baler/baler.py --mode compress --project profile cpu
python3 -m cProfile -o cProfile_decompress.prof  baler/baler.py --mode decompress --project profile cpu

#scalene
#the training profiling using scalene doesn't work
#scalene --cpu --gpu --memory baler/baler.py --mode train --project profile cpu
scalene --cpu --gpu --memory baler/baler.py --mode compress --project profile cpu
scalene --cpu --gpu --memory baler/baler.py --mode decompress --project profile cpu


#py-spy
py-spy top -- python3 baler/baler.py --mode train --project profile cpu
py-spy top -- python3 baler/baler.py --mode compress --project profile cpu
py-spy top -- python3 baler/baler.py --mode decompress --project profile cpu
