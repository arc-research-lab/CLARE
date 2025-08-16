# compose the source code into the image
# Use official Python image
FROM python:3.8.10-slim

# Set working directory
WORKDIR /CLARE

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything in repo
COPY . .
ENV PYTHONPATH=/CLARE:$PYTHONPATH

#enter the bash
CMD ["/bin/bash"]
