from django import forms
from .models import Task
import os


class TaskForm(forms.ModelForm):

    """
    Task form for uploading a file in the project page.
    """

    file = forms.FileField(label='file', widget=forms.FileInput(attrs={"id": "import-file", "accept": ".wav, .mp3, .mp4, .zip, .rar"}))
             
    class Meta:
        model = Task
        fields = [
            "file",
        ]
