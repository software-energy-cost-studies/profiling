# profiling
This repository contains the list of instructions and results for profiling of baler (data compression library):


### CPU profilers:

#### Profiling using cProfile
```
python3 -m cProfile -o cProfile_train.prof  baler.py --mode train --project profile cpu
python3 -m cProfile -o cProfile_train.prof  baler.py --mode compress --project profile cpu
python3 -m cProfile -o cProfile_train.prof  baler.py --mode decompress --project profile cpu
```

#### Scalene provides the easy to read the report about the CPU, GPU, memory, and counts (number of memory-expensive operations that appears during copying objects 
```
scalene --cpu --gpu --memory baler.py --mode train --project profile cpu
scalene --cpu --gpu --memory baler.py --mode compress --project profile cpu
scalene --cpu --gpu --memory baler.py --mode decompress --project profile cpu
```

#### py-spy
```
py-spy top -- python3 baler.py --mode train --project profile cpu
py-spy top -- python3 baler.py --mode compress --project profile cpu
py-spy top -- python3 baler.py --mode decompress --project profile cpu
```

### Run the torch profiler:
```
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)er
```



### GPU profilers:
1. [experiment-impact-tracker](https://github.com/Breakend/experiment-impact-tracker)
2. [scalene](https://github.com/plasma-umass/scalene)

### List of the framework for energy cost estimation:
1. [boagent](https://github.com/Boavizta/boagent)
2. [powermeter](https://github.com/autoai-incubator/powermeter)
3. [powerjoular](https://gitlab.com/joular/powerjoular)
4. [scaphandre](https://github.com/hubblo-org/scaphandre)
5. [AIPowerMeter](https://github.com/GreenAI-Uppa/AIPowerMeter)


### List of the framework for C02 emissions estimation:
1. [green-ai](https://github.com/daviddao/green-ai)
2. [Codecarbon](https://github.com/mlco2/codecarbon)
3. [Eco2AI](https://github.com/sb-ai-lab/Eco2AI)
4. [CarbonAI](https://github.com/Capgemini-Invent-France/CarbonAI)
5. [carbontracker](https://github.com/lfwa/carbontracker)
6. [tracarbon](https://github.com/fvaleye/tracarbon)
