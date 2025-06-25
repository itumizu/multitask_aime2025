#!/bin/bash

useradd -s /bin/bash -m ${USER_NAME} -p ${PASSWORD}
export HOME=/home/${USER_NAME}
usermod -u ${USER_ID} ${USER_NAME}
groupadd -g ${GROUP_ID} ${GROUP_NAME} 
usermod -g ${GROUP_NAME} ${USER_NAME}

chown ${USER_NAME}:${USER_NAME} -R /home/${USER_NAME}

mkdir -p /run/sshd /usr/sbin/sshd

sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sed "s@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g" -i /etc/pam.d/sshd

export VISIBLE=now >> /etc/profile
export PATH=/home/${USER_NAME}/.local/bin:/opt/conda/bin:$PATH >> /home/${USER_NAME}/.bashrc

echo "if [ -f ~/.bashrc ]; then  . ~/.bashrc;  fi" >> /home/${USER_NAME}/.bash_profile
echo "PYTHONPATH=${CURRENT_DIR_PATH}" >> /home/${USER_NAME}/.bash_profile
echo "JUPYTER_PATH=${CURRENT_DIR_PATH}" >> /home/${USER_NAME}/.bash_profile

echo "PATH=${PATH}:/usr/local/spark/bin" >> /home/${USER_NAME}/.bashrc
echo "cd ${CURRENT_DIR_PATH}" >> /home/${USER_NAME}/.bashrc
echo "JUPYTER_PATH=${CURRENT_DIR_PATH}" >> /home/${USER_NAME}/.bashrc

SITE_PACKAGE_PATH=`python -c "import site;print(site.getsitepackages()[0])"`
echo "${CURRENT_DIR_PATH}" > ${SITE_PACKAGE_PATH}/repo.pth

echo 'PATH="/usr/local/cuda/bin:$PATH"' >> /home/${USER_NAME}/.bashrc
echo 'LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> /home/${USER_NAME}/.bashrc

echo 'alias python=python3' >> /home/${USER_NAME}/.bashrc
echo 'alias pip=pip3' >> /home/${USER_NAME}/.bashrc

echo 'export PYTHONPATH='${CURRENT_DIR_PATH} >> /home/${USER_NAME}/.bashrc
echo 'export $(cat '${CURRENT_DIR_PATH}'/.env | grep -v "#" | xargs)' >> /home/${USER_NAME}/.bashrc

# Prefect configure
echo 'export PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://'$USER_NAME:'${POSTGRES_PASSWORD}@'$CONTAINER_NAME'_postgres:5432/prefect"' >> /home/${USER_NAME}/.bashrc
echo 'export POSTGRES_DB_URI="postgresql+psycopg2://'$USER_NAME:'${POSTGRES_PASSWORD}@'$CONTAINER_NAME'_postgres:5432"' >> /home/${USER_NAME}/.bashrc
echo 'export PREFECT_HOME='$CURRENT_DIR_PATH'/prefect' >> /home/${USER_NAME}/.bashrc
echo 'export PREFECT_UI_STATIC_DIRECTORY='$CURRENT_DIR_PATH'/prefect' >> /home/${USER_NAME}/.bashrc

# Git configure
echo 'git config --global user.name ${GITHUB_USERNAME}'>> /home/${USER_NAME}/.bashrc
echo 'git config --global user.email ${GITHUB_EMAIL}'>> /home/${USER_NAME}/.bashrc