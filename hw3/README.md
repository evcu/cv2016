## Traffic Sign Detection Kaggle Competition

For the results and report see my [ipython notebook](https://github.com/evcu/cv2016/blob/master/hw3/notebooks/Computer%20Vision%20HW3%20Report.ipynb)

In the last assignment of Computer Vision Class at 2016 Fall. We participated in a Kaggle competition with the rest of the class on the [The German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news). The objective was to produce a model that gives the highest possible accuracy on the test portion of this dataset. 

The benchmark is well researched and there are kwown architectures that gets below 1 percent error. I share in this post my experience with different Conv-Net architectures and Torch on HPC of NYU. 

I got 6th overall rank in the class out of 50+ people with 0.85 %Error. To run the code one use `main.lua` and one can see the `opts.lua` to what options exit. Models are defined at `models/` folder and preprocessing schemes are defined in `prepro/` folder. I exclude the data(images), which can be found [here](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset)

Over one week period I have experimented with different models on HPC. I've thought that limited GPU's are going to be pretty busy and therefore implement my code to work with cpu in a multithread matter. I've started by implementing preprocessing separete then the training, such that I don't preprocess same image twice. I've implemented ParallelIterator, however realized that it is not required when you already preprocessed the data and just reading from ram. Therefore haven't needed it for my experiments. I've follow a similar approach to organize my code `models/` file includes the models I have used and `prepro/` folder includes the preprocesing methods I've experimented. I've provided the convergence graphs of the models and the operation breakdowns with each model. I've used [Lua-profiler] to approximate #operations per model. (https://github.com/e-lab/Torch7-profiling/).
