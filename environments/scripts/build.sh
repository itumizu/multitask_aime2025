#!/bin/bash

USER_NAME=$(id -un)
USER_ID=$(id -u)
GROUP_NAME=$(id -gn)
GROUP_ID=$(id -g)
CURRENT_DIR_PATH=$(pwd)

echo `pwd`
source .env

if [ ! -e ./environments/etc/ssh/ssh_host_rsa_key ]; then
  ssh-keygen -t rsa -N '' -f ./environments/etc/ssh/ssh_host_rsa_key
fi

if [ ! -e ./environments/etc/ssh/ssh_host_dsa_key ]; then
  ssh-keygen -t dsa -N '' -f ./environments/etc/ssh/ssh_host_dsa_key
fi

if [ ! -e ./environments/etc/ssh/ssh_host_ed25519_key ]; then
  ssh-keygen -t ed25519 -N '' -f ./environments/etc/ssh/ssh_host_ed25519_key
fi

echo "------------------------------------------------------"
echo "1. ***** Generate SSH keys for use in container  *****"
echo ""

ssh-keygen -t ed25519 -f ./environments/etc/home/.ssh/id_ed25519_container
cp ./environments/etc/home/.ssh/id_ed25519_container.pub ./environments/etc/home/.ssh/authorized_keys

chmod 700 ./environments/etc/home/.ssh/
chmod 600 ./environments/etc/home/.ssh/id_ed25519_container
chmod 600 ./environments/etc/home/.ssh/authorized_keys

echo ""
echo "------------------------------------------------------"
echo "2. ***** Input user password *****"

# Hash salt
SALT=`python3 -c "import crypt; print(crypt.mksalt())"`

read -sp "Enter password for the user in the container.: " PASSWORD
tty -s && echo

hash_password=`python3 -c "import crypt, getpass, pwd, sys; print(crypt.crypt(sys.argv[1], sys.argv[2]))" $PASSWORD $SALT` 

# `openssl passwd -salt $salt $PASSWORD`
read -sp "Enter same password again for the user in the container.: " PASSWORD_REPEAT
hash_password_repeat=`python3 -c "import crypt, getpass, pwd, sys; print(crypt.crypt(sys.argv[1], sys.argv[2]))" $PASSWORD_REPEAT $SALT` 

is_same_hash=`python3 -c "import sys; from hmac import compare_digest; print(compare_digest(sys.argv[1], sys.argv[2]))" $hash_password $hash_password_repeat`

echo ""
echo "------------------------------------------------------"
echo "***** Build Docker image *****"
echo ""

if [ $is_same_hash ]
then
    echo "3. ***** Build PostgreSQL Docker image *****"

    ARR=(${IMAGE_TAG//:/ })
    IMAGE_TAG_POSTGERS=${ARR[0]}"_postgres:"${ARR[1]}
    echo $IMAGE_TAG_POSTGERS

    docker image build \
          ./environments/db_server \
          --tag ${IMAGE_TAG_POSTGERS} \

    echo "4. ***** Build Python Docker image *****"
    
    docker image build \
        ./environments/ \
        --tag ${IMAGE_TAG} \
        --build-arg USER_NAME=${USER_NAME} \
        --build-arg USER_ID=${USER_ID} \
        --build-arg GROUP_NAME=${GROUP_NAME} \
        --build-arg GROUP_ID=${GROUP_ID} \
        --build-arg CONTAINER_NAME=${CONTAINER_NAME} \
        --build-arg CURRENT_DIR_PATH=${CURRENT_DIR_PATH} \
        --build-arg PASSWORD=${hash_password}
      
    echo " ****** Build completed ******"

else
  echo "Passwords entered did not match. Try again."
  exit 1
fi