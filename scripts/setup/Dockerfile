#FROM centos:8

#RUN dnf install -y python39

#RUN pip install --upgrade pip

#RUN pip install python-dotenv && pip install azureml-core
ARG UBUNTU_VER=18.04
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
ARG PY_VER=3.9
ARG PANDAS_VER=1.3

FROM ubuntu:${UBUNTU_VER}
# System packages 
RUN apt-get update && apt-get install -yq curl wget jq vim

# Use the above args 
ARG CONDA_VER
ARG OS_TYPE
# Install miniconda to /miniconda
RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
RUN bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b
RUN rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda init

ARG PY_VER
ARG PANDAS_VER
# Install packages from conda 
RUN conda install -c anaconda -y python=${PY_VER}
RUN conda install -c anaconda -y \
    pandas=${PANDAS_VER}
