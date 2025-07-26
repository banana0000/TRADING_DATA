# Használjunk Python 3.10 image-et
FROM python:3.10-slim

# Munkakönyvtár létrehozása
WORKDIR /app

# Fájlok bemásolása
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Alkalmazás indítása
CMD ["python", "app.py"]

# Nyissuk ki a megfelelő portot (Render is ezt várja)
EXPOSE 8050