from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("upload/", views.upload, name="upload"),
    path("history/", views.history, name="history"),
    path("about/", views.about, name="about"),
]