# Useful commands for development process

## local
- **image_name:** mloptimizer
- **container_name:** mloptimizer

### create image from Dockerfile (execute it main project folder)
```
docker build -t mloptimizer .
```

### generate container from indicated image and mapping ports, make it in as daemon or do it in an interactive mode (executing "bash")
```
docker run -d --name mloptimizer mloptimizer
docker run -dp 127.0.0.1:8000:8501 --name mloptimizer mloptimizer
docker run -it --name mloptimizer mloptimizer bash
docker run -itdp 127.0.0.1:8000:8501 --name mloptimizer mloptimizer bash
```

### execute "bash" and get into the indicated container
```
docker exec -it mloptimizer bash
```

## official doc
- **image_name:** mloptimizer-img
- **container_name:** mloptimizer-app

### create image from Dockerfile (execute in main project folder)
```
docker build -t mloptimizer-img .
```

### generate container from indicated image and mapping ports, make it in as daemon or do it in an interactive mode (executing "bash")
```
docker run -d --name mloptimizer-app mloptimizer-img
docker run -dp 127.0.0.1:8000:8501 --name mloptimizer-app mloptimizer-img
docker run -it --name mloptimizer-app mloptimizer-img bash
docker run -itdp 127.0.0.1:8000:8501 --name mloptimizer-app mloptimizer-img bash
```

### execute "bash" and get into the indicated container
```
docker exec -it mloptimizer-app bash
```