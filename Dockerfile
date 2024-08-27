FROM python:3.9-slim

WORKDIR /app

# Install pipenv
RUN pip install pipenv

# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install dependencies
RUN pipenv install --system --deploy


# Create image directory
RUN mkdir -p /app/image

COPY . .

CMD ["python", "main.py"]