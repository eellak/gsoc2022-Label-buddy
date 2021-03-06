from django.contrib import admin

# Relative import
from .models import Project, Label


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


admin.site.register(Project, ProjectAmdin)
admin.site.register(Label, LabelAdmin)
