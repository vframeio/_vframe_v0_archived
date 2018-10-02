#!/bin/bash
#!/bin/bash
docker build --rm -t adamhrv/tesseract:ubuntu18 -f $a $(readlink -f $(dirname $0))/Dockerfile .
