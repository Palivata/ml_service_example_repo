# BarCode detection

his repository contains the service for model prediction based on images.

### Build
You need to have trained model from [here](https://github.com/Palivata/ml_example_repo):
```
dvc get https://github.com/Palivata/ml_example_repo models
```
To run the service locally:

```
make run_server_local
```

To run the service in Docker:

```
make build_server
make run_server_docker
```

To stop the service in Docker:
```
make stop_server
```

Running tests:


```
make run_tests
```

### Development
```
make initial
```
You can modify the code.


CI/CD:
To deploy: run job build
To stop server: run job destroy
