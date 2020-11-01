# -------- base ---------
FROM debian:latest AS base

RUN apt-get update && \
    apt-get install -y \
        wget \
        bash \
        build-essential \
        make \
        gcc \
        g++ \
        git \
        libncurses-dev \
        python3 \
        python3-numpy \
        python3-scipy \
        python3-matplotlib \
        python3-h5py \
        python3-yaml \
        python3-pytest \
        python3-pip \
        cython3 \
        jupyter \
        ipython3

RUN pip3 install --upgrade pip

# ------ NEURON -----------

FROM base AS neuron

RUN pip3 install neuron


# ----- LFPy ---------

FROM neuron AS lfpy

RUN apt-get update && \
    apt-get install -y \
    python3-mpi4py

RUN pip3 install git+https://github.com/LFPy/LFPy.git@2.2.dev0#egg=LFPy


# --- NEST ----

FROM lfpy AS nest

ARG WITH_MPI=ON
ARG WITH_OMP=ON
ARG WITH_GSL=ON

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    libgsl-dev \
    libltdl7 \
    libltdl-dev \
    libreadline7 \
    libreadline-dev \
    libboost-dev \
    doxygen

# Install NEST3 (master branch @24de43d)
RUN wget https://github.com/nest/nest-simulator/archive/24de43dc21c568e017839eeb335253c2bc2d487d.tar.gz && \
  mkdir nest-build && \
  tar zxf 24de43dc21c568e017839eeb335253c2bc2d487d.tar.gz && \
  mv nest-simulator-24de43dc21c568e017839eeb335253c2bc2d487d nest-simulator && \
  cd  nest-build && \
  cmake -DCMAKE_INSTALL_PREFIX:PATH=/opt/nest/ \
        -Dwith-boost=ON \
		-Dwith-ltdl=ON \
	    -Dwith-gsl=$WITH_GSL \
	    -Dwith-readline=ON \
        -Dwith-python=ON \
        -Dwith-mpi=$WITH_MPI \
        -Dwith-openmp=$WITH_OMP \
        ../nest-simulator && \
  make -j4 && \
  make install && \
  cd -

RUN echo "source /opt/nest/bin/nest_vars.sh" >> root/.bashrc

# clean up install/build files
RUN rm 24de43dc21c568e017839eeb335253c2bc2d487d.tar.gz
RUN rm -r nest*


# --- hybridLFPy ----

FROM nest AS hybridLFPy

RUN pip3 install git+https://github.com/INM-6/hybridLFPy.git@lfpy22#egg=hybridLFPy


# ---- hybridLFPy + examples ----

FROM hybridLFPy AS latest

RUN pip3 install git+https://github.com/NeuralEnsemble/parameters@b95bac2bd17f03ce600541e435e270a1e1c5a478
