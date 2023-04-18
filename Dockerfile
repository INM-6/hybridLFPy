FROM buildpack-deps:hirsute

RUN apt-get update && \
    apt-get install -y \
    cmake \
    libmpich-dev \
    mpich \
    doxygen \
    libboost-dev \
    libgsl-dev \
    cython3 \
    python3-dev \
    python3-pip \
    python3-numpy

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10

RUN pip install mpi4py

# Install NEST 3.4 (master branch @v3.4)
RUN git clone --depth 1 -b v3.4 https://github.com/nest/nest-simulator && \
  mkdir nest-build && \
  cd  nest-build && \
  cmake -DCMAKE_INSTALL_PREFIX:PATH=/opt/nest/ \
        -Dwith-boost=ON \
        -Dwith-ltdl=ON \
        -Dwith-gsl=ON \
        -Dwith-readline=ON \
        -Dwith-python=ON \
        -Dwith-mpi=ON \
        -Dwith-openmp=ON \
        ../nest-simulator && \
  make -j4 && \
  make install && \
  cd ..

# clean up install/build files
RUN rm -r nest-simulator
RUN rm -r nest-build


# ---- additional requirements
RUN apt-get install -y \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    python3-pandas \
    ipython3 \
    jupyter

RUN update-alternatives --install /usr/bin/ipython ipython /usr/bin/ipython3 10

# installing serial h5py (deb package installs OpenMPI which may conflict with MPICH)
RUN pip install h5py


# ---- install neuron -----
RUN apt-get install -y \
    bison flex

RUN git clone --depth 1 -b 8.2.2 https://github.com/neuronsimulator/nrn.git
RUN mkdir nrn-bld && cd nrn-bld

RUN cmake -DCMAKE_INSTALL_PREFIX:PATH=/opt/nrn/ \
  -DCURSES_NEED_NCURSES=ON \
  -DNRN_ENABLE_INTERVIEWS=OFF \
  -DNRN_ENABLE_MPI=ON \
  -DNRN_ENABLE_RX3D=OFF \
  -DNRN_ENABLE_PYTHON=ON \
  ../nrn

RUN cmake --build . --parallel 4 --target install && \
  cd ..

# add nrnpython to PYTHONPATH
ENV PYTHONPATH /opt/nrn/lib/python:${PYTHONPATH}


# --- Install hybridLFPy ----
RUN pip install git+https://github.com/INM-6/hybridLFPy.git@master#egg=hybridLFPy


# ---- Install additional dependencies for examples ----
RUN pip3 install git+https://github.com/NeuralEnsemble/parameters@b95bac2bd17f03ce600541e435e270a1e1c5a478


# Add NEST environment variables
RUN echo "source /opt/nest/bin/nest_vars.sh" >> root/.bashrc

# If running with Singularity, run the below line in the host.
# PYTHONPATH set here doesn't carry over:
# export SINGULARITYENV_PYTHONPATH=/opt/nest/lib/python3.9/site-packages
# Alternatively, run "source /opt/local/bin/nest_vars.sh" while running the container
