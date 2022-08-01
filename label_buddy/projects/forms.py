from django import forms
from .models import Project, PredictionModels


def get_model_tuple(model):

    tups = ()

    titles = model.objects.values('title')
    output_labels = model.objects.values('output_labels')
    model_ids = model.objects.values('id')
    models = model.objects.all()

    for model_id, title, output_label, model in zip(model_ids, titles, output_labels, models):
        
        title_str = str(title).split(':')[1].split('}')[0]
        output_label_str = str(output_label).split(':')[1].split('}')[0]
        model_id_str = str(model_id).split(':')[1].split('}')[0]

        new_entry = (model, f"Name: {title_str} | Labels: {output_label_str}")
        tups = (new_entry, ) + tups
    
    return tups


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
