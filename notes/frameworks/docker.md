# Docker 

#### Useful Docker commands

- list names of running containers: `docker ps`
- log int to docker: `docker exec -ti -u root container_name bash`
- reload daemon: `sudo systemctl daemon-reload`
- restart docker `sudo systemctl restart docker`

#### Change Image Storage Location

- Using many docker images can use several hundred GBs of store. It's often useful to move this off your statup disk
- Edit `sudo nano /etc/docker/daemon.json`
- Add
```
{
  "data-root": "/path/to/new/docker"
}
```
- stop docker `sudo systemctl stop docker`
- check docker has stopped `ps aux | grep -i docker | grep -v grep`
- copy data to new location `sudo rsync -axPS /var/lib/docker/ /path/to/new/docker`
- `sudo rsync -axPS /var/lib/docker/ /media/ubuntu/disk_name/data_store/docker_images` to copy to your new disk


#### Permissions

- permissions still not solved, but here are useful tips: <http://www.carlboettiger.info/2014/10/21/docker-and-user-permissions-crazyness.html>