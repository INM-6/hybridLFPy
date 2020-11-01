FROM debian:latest

RUN apt-get update && \
    apt-get install -y \
        wget \
        bash \
        build-essential \
        make \
        gcc \
        g++ \
        gfortran \
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


# ----- Install MPICH prepped for Singularity
ENV MPICH_DIR=/opt/mpich-3.3
ENV SINGULARITY_MPICH_DIR=$MPICH_DIR
ENV SINGULARITYENV_APPEND_PATH=$MPICH_DIR/bin
ENV SINGULAIRTYENV_APPEND_LD_LIBRARY_PATH=$MPICH_DIR/lib

ENV MPICH_VERSION=3.3
ENV MPICH_URL="http://www.mpich.org/static/downloads/$MPICH_VERSION/mpich-$MPICH_VERSION.tar.gz"
ENV MPICH_DIR=/opt/mpich

RUN mkdir -p /tmp/mpich
RUN mkdir -p /opt
RUN cd /tmp/mpich && wget -O mpich-$MPICH_VERSION.tar.gz $MPICH_URL && tar xzf mpich-$MPICH_VERSION.tar.gz
RUN cd /tmp/mpich/mpich-$MPICH_VERSION && ./configure --prefix=$MPICH_DIR && make install

ENV PATH=$MPICH_DIR/bin:$PATH
ENV LD_LIBRARY_PATH=$MPICH_DIR/lib:$LD_LIBRARY_PATH
ENV MANPATH=$MPICH_DIR/share/man:$MANPATH


# ------- Install NEURON and LFPy ----
RUN pip3 install --upgrade pip
RUN pip3 install mpi4py
RUN pip3 install neuron
RUN pip3 install git+https://github.com/LFPy/LFPy.git@2.2.dev0#egg=LFPy


# --- Install hybridLFPy ----
RUN pip3 install git+https://github.com/INM-6/hybridLFPy.git@nest3#egg=hybridLFPy


# ---- Install additional dependencies for examples ----
RUN pip3 install git+https://github.com/NeuralEnsemble/parameters@b95bac2bd17f03ce600541e435e270a1e1c5a478


# ------ Install NEST3 ----
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

# get number of cores
ENV NPROCS=$(getconf _NPROCESSORS_ONLN)

# Compile NEST3 (master branch @24de43d)
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
  make -j $NPROCS && \
  make install && \
  cd -

RUN echo "source /opt/nest/bin/nest_vars.sh" >> root/.bashrc

# clean up install/build files
RUN rm 24de43dc21c568e017839eeb335253c2bc2d487d.tar.gz
RUN rm -r nest*
