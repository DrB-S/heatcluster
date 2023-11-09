FROM ubuntu:jammy as app

ARG HEATCLUSTER_VER="0.4.12"

LABEL base.image="ubuntu:jammy"
LABEL dockerfile.version="1"
LABEL software.version="${HEATCLUSTER_VER}"
LABEL version="${HEATCLUSTER_VER}"
LABEL website="https://github.com/DrB-S/HeatCluster"
LABEL license="https://github.com/DrB-S/HeatCluster/blob/master/LICENSE"
LABEL name="heatcluster/${HEATCLUSTER_VER}"

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install pandas numpy pathlib seaborn matplotlib scipy 

# Make /data the working dir
RUN mkdir /data
WORKDIR /HeatCluster

COPY . .

ENV PATH="/HeatCluster-${HEATCLUSTER_VER}/bin:$PATH" \
 LC_ALL=C
 
CMD HeatCluster.py --help

WORKDIR /data

FROM app as test

# set working directory so that all test inputs & outputs are kept in /test
WORKDIR /test

# print help and version info; check dependencies (not all software has these options available)
# Mostly this ensures the tool of choice is in path and is executable
RUN heatcluster --help && \
 heatcluster --check && \
 heatcluster --version

# Demonstrate that the program is successfully installed

# Run the program's internal tests if available
RUN Heatcluster.py --test
