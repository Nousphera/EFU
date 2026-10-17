"""Native Flower client application."""

from flwr.clientapp import ClientApp
from .flower import handle

app = ClientApp()
app.train()(handle)
