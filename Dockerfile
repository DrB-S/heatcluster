FROM ubuntu:20.04

ARG HEATCLUSTER_VER="0.4.12"

LABEL base.image="ubuntu:20.04"
LABEL dockerfile.version="3"
LABEL software.version="${HEATCLUSTER_VER}"
LABEL version="${HEATCLUSTER_VER}"
LABEL website="https://github.com/DrB-S/HeatCluster"
LABEL license="https://github.com/DrB-S/HeatCluster/blob/master/LICENSE"
LABEL name="heatcluster/${HEATCLUSTER_VER}"

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install argparse pandas numpy pathlib seaborn matplotlib scipy 

WORKDIR /HeatCluster
COPY . .

ENV PATH=/HeatCluster:$PATH

CMD HeatCluster.py --help

WORKDIR /data
