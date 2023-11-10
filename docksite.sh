#!/bin/bash

usage(){ 
    echo "Usage: ./docksite.sh [options]" 
    echo "    [ -h | --help ]             : shows help" 
    echo "    [ --action start|stop ]     : start or stop the website - mandatory"
    echo "    [ -l | --log ]              : shows logs on output"
    echo "    [ -n | --name <string> ]    : lowercase name for docker container (default is dockersite). "
    echo "Example : ./docksite.sh --action start -l"
}

problem_build () {
    ret=$1
    if [ $ret -ne 0 ]; then
	echo "Problem trying to build Docker container (error code: $ret)"
	exit 1
    fi
}

problem_run () {
    ret=$1
    if [ $ret -ne 0 ]; then
	echo "Problem trying to run Docker container (error code: $ret)"
	exit 1
    fi
}

if [ $# -eq 0 ];
then
    usage
    exit 0
fi

CONTAINER_NAME=docksite

options=$(getopt -o lhn: --long log --long action: --long help --long name: -- "$@")
[ $? -eq 0 ] || {
    echo "Incorrect option provided"
    usage
    exit 1
}
eval set -- "$options"
while true; do
    case "$1" in
        -l)
            LOG=1
            ;;
	-h)
	    usage
	    exit 0
	    ;;
	-n)
	    shift
            CONTAINER_NAME=$1
            ;;
	--name)
	    shift
            CONTAINER_NAME=$1
            ;;
        --action)
	    shift
            ACTION=$1
            ;;
	--log)
	    LOG=1
	    ;;
	--help)
	    usage
	    exit 0
	    ;;
        --)
            shift
            break
            ;;
    esac
    shift
done

if [ -z $ACTION ]; then
    usage
    exit 1
fi

CONTAINER_NAME=$(echo "$CONTAINER_NAME" | tr '[:upper:]' '[:lower:]')

if [ $ACTION == start ]; then
    cd website

    if [ ! -z $LOG ]; then
	echo Building website
        docker build -t $CONTAINER_NAME .
	problem_build $?
	echo Launching website
	docker run -p 3000:3000 $CONTAINER_NAME &
	problem_run $?
    else
	echo Building website
	docker build -t $CONTAINER_NAME . > /dev/null 2>&1
	problem_build $?
	echo Launching website
	docker run -p 3000:3000 $CONTAINER_NAME > /dev/null 2>&1 &
	problem_run $?
    fi

    while [ -z "$DOCKER_ID" ]; do
	DOCKER_ID="$(docker ps  | grep $CONTAINER_NAME | awk '{print $1}')"
    done

    echo Website has been launched
    echo You can access the server on \"http://localhost:3000\"
    echo ID of the container $CONTAINER_NAME : \"$DOCKER_ID\"
    
elif [ $ACTION == stop ]; then
    echo Searching for container ID
    DOCKER_ID="$(docker ps  | grep $CONTAINER_NAME | awk '{print $1}')"
    
    if [ -z $DOCKER_ID ]; then
	echo There is nothing to stop
    else
	echo Stopping container $DOCKER_ID
	echo The container $(docker stop $DOCKER_ID) has been stopped
    fi
    
fi

exit 0
