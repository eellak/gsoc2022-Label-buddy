from django import forms
from .models import Project, PredictionModels


class ProjectForm(forms.ModelForm):

    """
    Project form for creating a project.
    """

    title = forms.CharField(label='Tile', required=False, widget=forms.TextInput(attrs={"placeholder": "Title"}))
    description = forms.CharField(required=False, widget=forms.Textarea(
        attrs={
            "placeholder": "Description",
            "rows": 4,
        }
    ))
    instructions = forms.CharField(required=False, widget=forms.Textarea(
        attrs={
            "placeholder": "Instructions",
            "rows": 4,
        }
    ))
    prediction_model = forms.ModelChoiceField(queryset=PredictionModels.objects.all())

    new_labels = forms.CharField(label="Labels", required=False, widget=forms.Textarea(
        attrs={
            "placeholder": "A comma separated list of new labels",
            "id": "new_labels",
            "rows": 4,
        }
    ))

    class Meta:
        model = Project
        fields = [
            "title",
            "description",
            "instructions",
            "logo",
            "prediction_model",
            "new_labels",
            "users_can_see_other_queues",
            "annotators",
            "reviewers",
            "managers",
        ]
