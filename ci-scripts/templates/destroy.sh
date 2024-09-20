#!/bin/bash

container_id=$(docker ps | grep registry.deepschool.ru/cvr7/k.polevoda/pr1_service/service2 | awk '{print $1}')

if [ ! -z "$container_id" ]; then
    docker stop $container_id
    docker rm $container_id
fi
