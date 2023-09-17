# MLOptimizer App
Interactive streamlit application to use the python library [mloptimizer](https://github.com/Caparrini/mloptimizer)

## Easy to install

### Prerequisites
Before you start, make sure you have [Docker](https://docs.docker.com/desktop) and, of course, your favorite IDE installed.

### Set up
1. Clone this repo to your local machine
2. Go to *mloptimizer-app* and build a Docker image from the Dockerfile provided:
```
cd ./mloptimizer-app/
docker build -t mloptimizer-img .
```
3. Run a new container and start using the app
```
docker run -dp 127.0.0.1:8000:8501 --name mloptimizer-app mloptimizer-img
```
Take into account that Streamlit app uses port 8501 of your new container, and it is mapped to your localhost 8000 port. You can edit command above to use a different port of your local machine.
4. You can now view the MLOptimizer App in your browser. Open your favorite browser and go to http://localhost:8000/

## Easy to use
Forget about huge python scripts, maintenance of libraries on your machine and difficult commands of a CLI. With this GUI, you can easily upload your CSV dataset and search for the best hyper-parameters for different cases. You'll be able, for example, to choose between different algorithms, specify amount of generations and individuals, set ranges of values for the hyper-paramaters or even decide which ones you want to keep fixed.
