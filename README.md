# Please define the absolute path of the folder that you want virtuoso to map to, from there you will upload the files to virtuoso
export ABS_PATH=/media/apc/DATA1/Maastricht

# run this command to run Virtuoso, it will also install it if it is not there

sudo docker run --rm --name virtuoso.2.7.1 \
    -p 8890:8890 -p 1111:1111 \
    -e DBA_PASSWORD=dba \
    -e SPARQL_UPDATE=true \
    -e DEFAULT_GRAPH=http://bio2rdf.com/bio2rdf_vocabulary: \
    -v ${ABS_PATH}/bio2rdf-github/virtuoso:/data \
    -d tenforce/virtuoso:1.3.2-virtuoso7.2.1

sudo docker build -t favlib .

sudo docker run -d --rm --link virtuoso.2.7.1:virtuoso.2.7.1 --name favlib -p 8888:8888 -v /media/apc/DATA1/Maastricht/favlib:/jupyter -v /tmp:/tmp favlib

sudo docker exec -it favlib cwltool --outdir=/jupyter/output/ workflow/main-workflow.cwl workflow/workflow-job.yml
