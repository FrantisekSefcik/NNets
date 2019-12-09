# Neural network - GAN project

#### František Šefčík, Vajk Pomichal

### Instalation

1. Clone repository
```shell script
git clone git@github.com:FrantisekSefcik/NNets.git
```

2. Build docker image
```shell script
cd NNets/text_to_image
docker build -t nnets/tensorflow:2.0.0-gpu-py3-jupyter .
```

3. Run docker container
```shell script
cd ..
docker run --gpus all -u $(id -u):$(id -g) --rm -p 8888:8888 -p 6006:6006 -v $(pwd):/project -it --name nnets_project nnets/tensorflow:2.0.0-gpu-py3-jupyter

```

## Project documentation

You can find project documentation in 
[/documents/Documentation](https://github.com/FrantisekSefcik/NNets/blob/master/documents/Documentation.md). 
There is described architecture of model, learning pipeline and evaluation of experiments.  

