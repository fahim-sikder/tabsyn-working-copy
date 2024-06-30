#!/bin/bash

docker run --gpus all -it --rm \
	-v /home/$USER:/home/$USER \
	tabsyn \
	/bin/bash -c "cd /home/$USER; exec /bin/bash"
