FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04

WORKDIR /work
ENV DEBIAN_FRONTEND=noninteractive

ARG USER_ID
ARG USER_NAME
ARG GROUP_ID
ARG GROUP_NAME
ARG PASSWORD
ARG CURRENT_DIR_PATH
ARG CONTAINER_NAME

ENV LANG C.UTF-8
ENV USER_ID $USER_ID
ENV USER_NAME $USER_NAME
ENV GROUP_ID $GROUP_ID
ENV GROUP_NAME $GROUP_NAME
ENV PASSWORD $PASSWORD
ENV CURRENT_DIR_PATH $CURRENT_DIR_PATH
ENV CONTAINER_NAME $CONTAINER_NAME

ENV NOTVISIBLE "in users profile"

RUN apt-get update
RUN apt-get install -y apt-utils

RUN apt update && apt upgrade -y && apt install -y build-essential ca-certificates locales-all \
    automake pkg-config libevent-dev libncurses5-dev bison \
    libncursesw5-dev libssl-dev libsqlite3-dev libgdbm-dev libbz2-dev graphviz \
    liblzma-dev zlib1g zlib1g-dev uuid-dev libffi-dev libdb-dev bzip2 checkinstall libreadline-gplv2-dev tk-dev \
    software-properties-common git emacs vim curl unzip htop openssh-server wget procps sudo nodejs postgresql-client 

RUN apt-get autoremove -y

# tmux install
RUN git clone https://github.com/tmux/tmux.git .  \
    && git checkout 3.4 \
    && sh autogen.sh \
    && ./configure \
    && make \
    && make install

# Python install
RUN wget https://www.python.org/ftp/python/3.11.5/Python-3.11.5.tgz \
    && tar -xf Python-3.11.5.tgz \
    && cd Python-3.11.5 \
    && ./configure --enable-optimizations\
    && make \
    && make install

COPY scripts/entrypoint.sh /var/tmp
RUN chmod a+x /var/tmp/entrypoint.sh
RUN /var/tmp/entrypoint.sh

COPY scripts/extra.sh /var/tmp
RUN chmod a+x /var/tmp/extra.sh
RUN /var/tmp/extra.sh

RUN usermod -G sudo ${USER_NAME}
RUN /usr/bin/ssh-keygen -A

RUN mkdir -p $CURRENT_DIR_PATH
WORKDIR $CURRENT_DIR_PATH

COPY ./etc/home/.ssh/ /home/$USER_NAME/.ssh
COPY python/requirements.base.txt $CURRENT_DIR_PATH
COPY python/requirements.txt $CURRENT_DIR_PATH

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.base.txt
RUN pip3 install -r requirements.txt

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]