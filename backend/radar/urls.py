from django.urls import path
from . import views

urlpatterns = [
    path("metadata/", views.metadata),
    path("latest.png", views.latest_png),
]