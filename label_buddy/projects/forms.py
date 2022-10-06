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
    prediction_model = forms.ModelChoiceField(queryset=PredictionModels.objects.all(), empty_label="No model.", required=False, widget=forms.Select(attrs={"id": "prediction_model",}))
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


class PredictionModelForm(forms.ModelForm):

    """
    Prediction Model form for adding a prediction model.
    """

    title = forms.CharField(label='Tile', required=True, widget=forms.TextInput(attrs={"placeholder": "Title"}))
    output_labels = forms.CharField(label="Labels", required=True, widget=forms.Textarea(
        attrs={
            "placeholder": "Prediction Model output labels",
            "rows": 4,
        }
    ))
    image_repo = forms.CharField(label='Image Repo', required=False, widget=forms.TextInput(attrs={"placeholder": "image_repo"}))
    weight_file = forms.FileField(label="Model File", required=False, widget=forms.FileInput(attrs={"id": "weight_file"}))
    test_dataset = forms.FileField(label="Test Dataset", required=False, widget=forms.FileInput(attrs={"id": "test_dataset"}))
    current_accuracy_precentage = forms.FloatField(label="Current Accuracy", required=False, widget=forms.NumberInput(attrs={"id": "current_accuracy_precentage"}))
    current_loss_precentage = forms.FloatField(label="Current Loss", required=False, widget=forms.NumberInput(attrs={"id": "current_loss_precentage"}))

    class Meta:
        model = PredictionModels
        fields = [
            "title",
            "output_labels",
            "image_repo",
            "weight_file",
            "test_dataset",
            "current_accuracy_precentage",
            "current_loss_precentage"
        ]

