FROM python:3.9

# Définir le répertoire de travail dans /app
WORKDIR /app

# Copier les fichiers locaux dans le conteneur
COPY . /app

# Installer les dépendances
RUN pip install -r /app/requirements.txt

# Exposer le port utilisé par l'API
EXPOSE 5000

# Commande pour démarrer l'application
CMD ["python", "app.py"]



