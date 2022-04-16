cd /home/user/projects/imagenet_tutorial/jetson-inference &&
docker/run.sh --volume ~/my-detection-python:/jetson-inference/my-detection-python --container=b02828e87f05 --run my-detection-python/camera_based_person_counter_API2_with_search.py
