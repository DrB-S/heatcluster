ARG HEATCLUSTER_VER="0.4.12"

FROM ubuntu:jammy as app
USER root

# re-instantiating for the app build layer if using ARG as a global variable above
ARG HEATCLUSTER_VER

LABEL base.image="ubuntu:jammy"
LABEL dockerfile.version="3"
LABEL software="HeatCluster"
LABEL software.version="${HEATCLUSTER_VER}"
LABEL description="This software produces a heatmap for a SNP matrix"
LABEL website="https://github.com/DrB-S/HeatCluster"
LABEL license="https://github.com/DrB-S/HeatCluster/blob/master/LICENSE"
LABEL name="heatcluster/${HEATCLUSTER_VER}"
LABEL maintainer="Stephen Beckstrom-Sternberg"
LABEL maintainer.email="stephen.beckstrom-sternberg@azdhs.gov"

# 'RUN' executes code during the build
# Install Python and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
/usr/bin/python3 python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install pandas numpy pathlib seaborn matplotlib scipy 

ENV PATH="$PATH"

# Set /data as working dir
RUN mkdir /data
WORKDIR /data

RUN echo "installing heatcluster" && echo

COPY . .

RUN echo "Change to lowercase name: " && \
mv HeatCluster-${HEATCLUSTER_VER} heatcluster-${HEATCLUSTER_VER}
RUN chmod +x bin/heatcluster

# 'ENV' instructions set environment variables that persist from the build into the resulting image
# Use for e.g. $PATH and locale settings for compatibility with Singularity
ENV PATH="/heatcluster-${HEATCLUSTER_VER}/bin:$PATH" \
 LC_ALL=C

# 'CMD' instructions set a default command when the container is run.  
CMD HeatCluster.py --help

FROM app as test

# set working directory so that all test inputs & outputs are kept in /test
#WORKDIR /test

# print help and version info; check dependencies (not all software has these options available)
# Mostly this ensures the tool of choice is in path and is executable
#RUN heatcluster --help && \
# heatcluster --check && \
# heatcluster --version

# Demonstrate that the program is successfully installed

RUN echo && echo "Show heatcluster help file:  " && echo && \
 heatcluster -h  && echo
 
 RUN echo && echo "Run a test matrix thru the program" && \
python3 HeatCluster.py -i snp-dists.txt && echo
