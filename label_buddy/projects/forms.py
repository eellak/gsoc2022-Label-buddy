from django import forms
from .models import Project, PredictionModels


def get_model_tuple(model):

    tups = ()

    titles = model.objects.values('title')
    output_labels = model.objects.values('output_labels')

    counter = 1
    for title, output_label in zip(titles, output_labels):
        
        title_str = str(title).split(':')[1].split('}')[0]
        output_label_str = str(output_label).split(':')[1].split('}')[0]

        new_entry = (str(counter), f"Name: {title_str} - Labels: {output_label_str}")
        tups = (new_entry, ) + tups
        counter += 1

    print(tups)

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
    prediction_model = forms.ChoiceField(choices=get_model_tuple(PredictionModels), widget=forms.Select(attrs={"id": "prediction_model",}))
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
