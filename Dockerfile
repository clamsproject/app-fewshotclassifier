# Use the same base image version as the clams-python python library version
FROM clamsproject/clams-python:x.y.z
# See https://hub.docker.com/search?q=clams-python for more base images

################################################################################
# clams-python base images are based on debian distro
# install more system packages as needed using the apt manager
################################################################################

################################################################################
# main app installation
COPY ./ /app
WORKDIR /app
RUN pip3 install -r requirements.txt

# default command to run the CLAMS app in a production server 
CMD ["python3", "app.py", "--production"]
################################################################################
