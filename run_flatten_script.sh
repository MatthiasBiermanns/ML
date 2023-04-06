#!/bin/bash

ROOT=$(pwd)
echo ${ROOT}

runscript()
{
    cd database
    /bin/python3 ${ROOT}/script.py
    cd ../query
    /bin/python3 ${ROOT}/script.py
    pwd
}


# Flatten Dataset in Berlin
cd berlin
pwd
runscript
## Back to Root
cd ${ROOT}
pwd

# Flatten Dataset in Kampala
cd kampala
pwd
runscript
## Back to Root
cd ${ROOT}
pwd

# Flatten Dataset in Melbourne
cd melbourne
pwd
runscript
## Back to Root
cd ${ROOT}
pwd

# Flatten Dataset in Sao Paulo
cd saopaulo
pwd
runscript
## Back to Root
cd ${ROOT}
pwd

# Flatten Dataset in San Francisco
cd sf
pwd
runscript
## Back to Root
cd ${ROOT}
pwd
# Flatten Dataset in Tokyo
cd tokyo
pwd
runscript