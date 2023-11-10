#!/bin/bash

usage(){ 
    echo "Usage: ./launch_website.sh [options]" 
    echo "    [ -h | --help ]             : shows help" 
    echo "    [ --action start|stop ]     : start or stop the website"
    echo "    [ -l | --log ]              : shows logs on output"
    echo "Example : ./launch_website.sh --action start -l"
}

if [ $# -eq 0 ];
then
    usage
    exit 0
fi

options=$(getopt -o lh --long log --long action: --long help -- "$@")
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

if [ $ACTION == start ]; then
    cd website

    if [ ! -z $LOG ]; then
	echo Building website
        docker build -t my_website .
	echo Launching website
	docker run -p 3000:3000 my_website &
	sleep 5
	echo Website has been launched
	echo You can access the server on \"http://localhost:3000\"
    else
	echo Building website
	docker build -t my_website . > /dev/null 2>&1
	echo Launching website
	docker run -p 3000:3000 my_website > /dev/null 2>&1 &
        sleep 5
	echo Website has been launched
	echo You can access the server on \"http://localhost:3000\"
    fi
    
elif [ $ACTION == stop ]; then
    echo seatching for docker
    DOCKER_ID="$(docker ps  | grep my_website | awk '{print $1}')"
    echo $DOCKER_ID
    docker stop $DOCKER_ID
fi

exit 0
