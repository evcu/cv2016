# torch7-pruner : Prunning Convolutional Networks

This is my final project for the Computer Vision class thought by Rob Fergus in Fall 2016. I got the project of implementing compressing ideas of Song Han presented in paper `Learning both Weights and Connections for Efficient Neural Networks`. Brief summary of my-work and results can be found [here](https://evcu.github.io/)

## Quick Start
Clone the repo
```
git clone https://github.com/evcu/cv2016.git
cd cv2016/project/
mkdir logs/
```
Ensure that you have torch library. One can run `module load torch/gnu/20160623` to get environment set in NYU's HPC Mercer.

There is one pretrained `lenet5.t7` model provided in `inp/` folder. One can use `qlua train.lua` command and its various flags to train a new model, but we will use the provided pretrained model.

One can perform layer-wise prunning with `-l` flag represents the layar-index in the model provided and `-p` represents the percentage to be prunned. One should get a result similar to.
```
$ qlua main.lua -l 8 -p 0.7
....
Layer8: 30% retained
 [============== 79/79 =============>] Tot: 9s735ms | Step: 124ms    
nil: avg. loss: 0.0301; avg. error: 0.9800, time: 9.8613
Total Compression: 0.64312374389894
```

If retraining after prunning enabled with binary flag `-reTrain`. `-nEpochs` is used to set the total retraining epochs. 

`qlua main.lua -reTrain -nEpochs 3 -l 8 -p 0.7`

Iterative prunning enabled with providing `-iPrunning` with a non-1 value. In the scenario below at each iteration 7% of layer's parameters are going to be prunned and 3 retraining epochs are going to be performed. This is gonna be repeated 10 times.  

`qlua main.lua -iPrunning 10 -reTrain -nEpochs 3 -l 8 -p 0.7`

One can generate data for sensitivity plots with iterative prunning, since the results at each prunning iterations are saved to `logs/` folder. Use `-reLoad` flag to reload each model ath each prunning iteration such that one-time prunning is performed for each intermediate prunning factors.

`qlua main.lua -reLoad -iPrunning 10 -reTrain -nEpochs 3 -l 8 -p 0.7`

One can try various pruner functions with `-pruner` flag, which has the following options.
- Taylor series based approximations of $\delta E$: 
    - Using 1st order approximation: `-pruner taylor1`
    - Using 2nd order diagonal approximation: `-pruner taylor2`
    - Combining these two `-pruner taylor12`
- Regularization based learning methods, where the original weights are multiplied with a constant (initialy 1) and then those constant factors are learned through regularized cost function. Connections with smaller weights are supposed to be less important.
    - L1 based `-pruner l1`
    - L2 based `-pruner l2`
- Emprical measure, calculated by pruning each weight one by one and calculating test error for each weight. Then the weights are pruned in the reverse order `-pruner emp`

To enable GPU implementation use `-cuda` flag.

To prune different layers together one need to provide a `inp/*.conf`  file with the following CSV format
```
1,4,8,10,12
0.5,0.7,0.95,0.7,0.5
```
First line consists of layer-indices to be pruned(`-l`), whereas the second line is the prunning factors (`-p`). 

If no `-l` flag is provided script automatically loads the configiration provided. So one can prune whole network
- iteratively `qlua main.lua -iPrunning 10 -reTrain -nEpochs 3`
- one-shot `qlua main.lua -reTrain -nEpochs 3`
- without-retraining `qlua main.lua`

To test a pruned network one can use `-test` flag. A pruned versio of lenet-5 provided in the `out/` folder and one can test it with

```
qlua main.lua -test
...
Total Compression: 0.92905541200115
Error: 0.94
```

One can use `-acctradeoff` flag to stop prunning individual layers if accuracy decreases more then the value provided. For example for a value of `0.5` prunning of individual layers would stop when the accuracy-loss after training is more then 0.5% 

`qlua main.lua -acctradeoff 0.5 -iPrunning 10 -reTrain -nEpochs 3`

