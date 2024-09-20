#!/bin/bash

export PWD="$1"
export USER="$2"
export REGISTRY="$3"

echo $PWD | docker login -u $USER  $REGISTRY --password-stdin
docker pull $REGISTRY/cvr7/$USER/pr1_service/service2

docker run -d -p 5053:5053 $REGISTRY/cvr7/$USER/pr1_service/service2
