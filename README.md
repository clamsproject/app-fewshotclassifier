# app-few-shot-classifier

This repository provides a wrapper for using CLIP embeddings for few shot classification.

1. `app.py` to write the app 
1. `requirements.txt` to specify python dependencies
1. `Containerfile` to containerize the app and specify system dependencies
1. an empty `LICENSE` file to replace with an actual license information of the app
1. this `README.md` file with basic instructions of app installation and execution
1. some GH actions workflow for issue/bug-report management
1. a GH actions workflow to build and upload app images upon any push of a git tag

Modify this file as needed to provide proper instructions for your users. 

## Requirements 

Generally, an CLAMS app requires 

- Python3 with the `clams-python` module installed; to run the app locally. 
- `docker`; to run the app in a Docker container (as a HTTP server).
- A HTTP client utility (such as `curl`); to invoke and execute analysis.

## Building and running the Docker image

From the project directory, run the following in your terminal to build the Docker image from the included Dockerfile:

```bash
docker build . -f Dockerfile -t <app_name>
```

Alternatively, the app maybe already be available on docker hub. 

``` bash 
docker pull <app_name>
```

Then to create a Docker container using that image, run:

```bash
docker run -v /path/to/data/directory:/data -p <port>:5000 <app_name>
```

where /path/to/data/directory is the location of your media files or MMIF objects and <port> is the *host* port number you want your container to be listening to. The HTTP inside the container will be listening to 5000 by default. 

## Invoking the app
Once the app is running as a HTTP server, to invoke the app and get automatic annotations, simply send a POST request to the app with a MMIF input as request body.

MMIF input files can be obtained from outputs of other CLAMS apps, or you can create an empty MMIF only with source media locations using `clams source` command. See the help message for a more detailed instructions. 

```bash
clams source --help
```

(Make sure you installed the same `clams-python` package version specified in the [`requirements.txt`](requirements.txt).)

