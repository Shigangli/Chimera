# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess

DOCKER_IMAGE_NAME = None  # TODO: Replace with image name.
REPOSITORY_NAME = None    # TODO: Replace with repository name.

class WorkerInfo(object):
    def __init__(self, ip, port=22, internal_ip=None):
        self.ip = ip
        self.port = port
        self.internal_ip = internal_ip

    def __repr__(self):
        return '%s:%s' % (self.ip, self.port)


def read_workers_file(filename):
    workers = []
    with open(filename, 'r') as f:
        for line in f:
            worker = line.strip()
            worker_info = worker.split(":")
            assert len(worker_info) == 2 or len(worker_info) == 3, worker
            internal_ip = None
            if len(worker_info) == 3:
                internal_ip = worker_info[2]
            workers.append(WorkerInfo(ip=worker_info[0],
                                      port=worker_info[1],
                                      internal_ip=internal_ip))
    return workers

def update_all(workers):
    for worker in workers:
        node_ip = worker.ip
        node_port = worker.port
        mount_directories = ['/home/ubuntu']
        docker_cmd = 'nvidia-docker run %(mount_directories)s ' \
                     '--net=host ' \
                     '--ipc=host %(docker_image_name)s /bin/bash -c \'%(command)s\'' % {
            "docker_image_name": DOCKER_IMAGE_NAME,
            "mount_directories":
                " ".join(["-v %s:%s" % (x, x)
                          for x in mount_directories]),
            "command": "cd %s; git fetch origin; git checkout master; git reset --hard origin/master" % REPOSITORY_NAME
        }
        subprocess.call(
            "ssh -n %s -p %s -o StrictHostKeyChecking=no \"%s\"" % (
                node_ip, node_port, docker_cmd),
            shell=True)


if __name__ == '__main__':
    workers = read_workers_file('workers.txt')
    update_all(workers)
