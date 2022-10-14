from django.contrib import admin

# Relative import
from .models import Project, Label, PredictionModels


# Admin now has filters and search
class ProjectAmdin(admin.ModelAdmin):

    """
    Project class for the admin site. list_display shows the fields
    displayed in the admin site.
    """

    search_fields = ["title"]
    list_display = ["id", "title", "created_at", "users_can_see_other_queues", "project_type"]
    ordering = ("created_at",)
    list_filter = ["users_can_see_other_queues", "project_type"]


class LabelAdmin(admin.ModelAdmin):

    """
    Label class for the admin site. list_display shows the fields
    displayed in the admin site.
    """

    list_display = ["name", "parent"]


class PredictionModelsAdmin(admin.ModelAdmin):

    """
    PredictionModels class for the admin site. list_display shows the fields
    displayed in the admin site.
    """

    search_fields = ["title"]
    list_display = ["id", "title", "output_labels", "docker_configuration_yaml_file", "weight_file", "current_accuracy_precentage"]
    ordering = ("id",)
    list_filter = ["output_labels", "current_accuracy_precentage"]


admin.site.register(Project, ProjectAmdin)
admin.site.register(Label, LabelAdmin)
admin.site.register(PredictionModels, PredictionModelsAdmin)
