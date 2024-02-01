FROM ubuntu:jammy as app

# default version
ARG HEATCLUSTER_VER="1.2.0.20240201"

# adding labels
LABEL base.image="ubuntu:jammy"
LABEL dockerfile.version="1"
LABEL software="heatcluster"
LABEL software.version="${HEATCLUSTER_VER}"
LABEL version="${HEATCLUSTER_VER}"
LABEL description="Visualize SNP matrix from snp-dists output"
LABEL website="https://github.com/DrB-S/heatcluster"
LABEL license="https://github.com/DrB-S/heatcluster/blob/master/LICENSE"
LABEL name="heatcluster/${HEATCLUSTER_VER}"
LABEL maintainer="Stephen Beckstrom-Sternberg"
LABEL maintainer.email="stephen.beckstrom-sternberg@azdhs.gov"

# installing apt-get dependencies
RUN apt-get update && apt-get upgrade -y && \
  apt-get install -y --no-install-recommends \
  ca-certificates \
  wget \
  procps \
  python3 \
  python3-pip && \
  apt-get autoclean && rm -rf /var/lib/apt/lists/*

# installing python dependencies
RUN pip3 install --no-cache argparse pandas numpy pathlib seaborn seaborn_polars polars pyarrow matplotlib scipy --upgrade-strategy=only-if-needed

# copying files to docker image
COPY . /heatcluster

ENV PATH=/heatcluster:$PATH

# makes sure heacluster is in path
RUN heatcluster.py -h

# default command for the container
CMD heatcluster.py -h

FROM app as test

WORKDIR /test

RUN echo "Show heatcluster version number and help file:  " && \
  heatcluster.py --version && \
  heatcluster.py --help

RUN echo "Test a small and medium matrix :" && \
  heatcluster.py -i /heatcluster/test/small_matrix.csv -t png -o small_test && \
  heatcluster.py -i /heatcluster/test/med_matrix.txt -t pdf -o med_test && \
  ls med_test.pdf && ls small_test.png

