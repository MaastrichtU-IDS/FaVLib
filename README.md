## How to run

## Using Docker 
* First make sure Docker is installed!!

* Clone the current repository
```shell
git clone https://github.com/MaastrichtU-IDS/FaVLib.git
```
* Enter the repository
```shell
cd FavLib
```

```shell
docker build -t favlib .
```

```shell
docker run -d  --rm --name favlib -p 8888:8888 -v $(pwd):/jupyter -v /tmp:/tmp favlib
```
```shell
docker exec -it favlib cwltool --outdir=/jupyter/output/ workflow/main-workflow-pykeen.cwl workflow/workflow-pykeen.yml
```

* That's it!!
