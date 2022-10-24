from django.http import HttpResponse
from django.utils import timezone
from django.contrib import messages

from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.exceptions import PermissionDenied
from rest_framework import (
    permissions,
    status,
)

from projects.models import Project
from .models import Task, Annotation
from .serializers import TaskSerializer
from .helpers import (
    get_user,
    get_annotation,
    export_data,
    format_exported_json
)

from zipfile import ZipFile
import io
import numpy as np
import os
# Create your views here


# API VIEWS
class TaskList(APIView):

    permission_classes = (permissions.IsAuthenticatedOrReadOnly,)
    serializer_class = TaskSerializer

    """
    List all tasks or create a new one.
    """

    # Get request
    def get(self, request, format=None):

        tasks = Task.objects.all()
        serializer = TaskSerializer(tasks, many=True)

        return Response(serializer.data)

    # Post request
    def post(self, request, format=None):

        serializer = TaskSerializer(data=request.data)

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class AnnotationSave(APIView):

    """
    Only post is implemented here.
    """

    def get_project(self, pk):
        try:
            return Project.objects.get(pk=pk)
        except PermissionDenied:
            return Response({"message": "No permissions"}, status=status.HTTP_401_UNAUTHORIZED)
        except Project.DoesNotExist:
            return Response({"message": "Project does not exist"}, status=status.HTTP_400_BAD_REQUEST)

    def get_task(self, task_pk):
        try:
            return Task.objects.get(pk=task_pk)
        except PermissionDenied:
            return Response({"message": "No permissions"}, status=status.HTTP_401_UNAUTHORIZED)
        except Task.DoesNotExist:
            return Response({"message": "Task does not exist"}, status=status.HTTP_400_BAD_REQUEST)

    def post(self, request, pk, task_pk, format=None):

        # Checks for user
        user = get_user(request.user.username)
        if not user or (user != request.user):
            return Response({"message": "Something is wrong with the user!"}, status=status.HTTP_400_BAD_REQUEST)

        # Check if project exists
        project = self.get_project(pk)
        if isinstance(project, HttpResponse):
            return Response(project.data, status=project.status_code)

        # Check if task exists
        task = self.get_task(task_pk)
        if isinstance(task, HttpResponse):
            return Response(task.data, status=task.status_code)

        # Check if user is annotator for this project
        if user not in project.annotators.all():
            return Response({"message": "You are not an annotator for this project!"}, status=status.HTTP_400_BAD_REQUEST)

        # Check if task belongs to project
        if task.project != project:
            message = "Task " + str(task.id) + " does not belong to project " + project.title + "!"
            return Response({"message": message}, status=status.HTTP_400_BAD_REQUEST)

        # If all validations pass, save annotations
        # Check if an annotation already exists. If not create one, else change result and update date
        annotation = get_annotation(task, project, user)
        result = request.data

        # If annotation is not empty
        if annotation:
            if result != []:
                # Update existing annotation
                annotation.result = result
                annotation.updated_at = timezone.now()
                annotation.save()
            else:
                annotation.delete()
                messages.add_message(request, messages.SUCCESS, "Annotation deleted.")
                return Response({}, status=status.HTTP_200_OK)
        else:
            if result != []:
                # Create new annotation
                Annotation.objects.create(
                    task=task,
                    project=project,
                    user=user,
                    result=result,
                )
            else:
                messages.add_message(request, messages.ERROR, "You submitted an empty annotation.")
                return Response({}, status=status.HTTP_400_BAD_REQUEST)
        messages.add_message(request, messages.SUCCESS, "Annotation saved successfully.")
        return Response({}, status=status.HTTP_200_OK)


class ExportData(APIView):

    """
    API endpoint for exporting data for a project.
    Only post is implemented here.
    """

    def get_project(self, pk):
        try:
            return Project.objects.get(pk=pk)
        except PermissionDenied:
            return Response({"message": "No permissions"}, status=status.HTTP_401_UNAUTHORIZED)
        except Project.DoesNotExist:
            return Response({"message": "Project does not exist"}, status=status.HTTP_400_BAD_REQUEST)

    def post(self, request, pk, format=None):

        # Checks for user
        user = get_user(request.user.username)
        if not user or (user != request.user):
            return Response({"message": "Something is wrong with the user!"}, status=status.HTTP_400_BAD_REQUEST)

        # Check if project exists
        project = self.get_project(pk)
        if isinstance(project, HttpResponse):
            return Response(project.data, status=project.status_code)

        # Check if user is part of this project
        if (user not in project.reviewers.all()) and (user not in project.annotators.all()) and (user not in project.managers.all()):
            return Response({"message": "You are not involved to this project!"}, status=status.HTTP_400_BAD_REQUEST)

        # If all validations pass, return exported json
        exported_json, skipped_annotations, audios = export_data(project, request.data['exportApproved'])

        # Create filename
        if request.data['exportApproved']:
            exported_name = "project-" + str(project.id) + "-ONLY_APPROVED-export_at-" + project.created_at.strftime("%Y-%m-%d-%H:%M:%S")
        else:
            exported_name = "project-" + str(project.id) + "-export_at-" + project.created_at.strftime("%Y-%m-%d-%H:%M:%S")

        if request.data['exportApproved']:
            message = str(skipped_annotations) + " unapproved annotations were skipped. Succesfully exported " + exported_name + "."
        else:
            message = "Succesfully exported " + exported_name + "."
        exported_name += ".json" if request.data['format'] == "JSON" else ".csv"
        data = {
            "format": request.data['format'],
            "exported_json": exported_json,
            "exported_name": exported_name,
            "message": message
        }
        return Response(data, status=status.HTTP_200_OK)


class ExportDataToContainer(APIView):

    """
    API endpoint for exporting data for a project to a container for the training/validation process.
    Only post is implemented here.
    """

    def get_project(self, pk):
        try:
            return Project.objects.get(pk=pk)
        except PermissionDenied:
            return Response({"message": "No permissions"}, status=status.HTTP_401_UNAUTHORIZED)
        except Project.DoesNotExist:
            return Response({"message": "Project does not exist"}, status=status.HTTP_400_BAD_REQUEST)

    def post(self, request, pk):

        # Check if project exists
        project = self.get_project(pk)
        if isinstance(project, HttpResponse):
            return Response(project.data, status=project.status_code)

        # If all validations pass, return exported json
        exported_json, skipped_annotations, audios = export_data(project, True)

        final_annotations = format_exported_json(exported_json)
        final_annotations_path = f"./media/data/temp_project{pk}_annotations.npy"
        np.save(final_annotations_path, final_annotations)

        # create zip file with audios and annotations
        buffer = io.BytesIO()
        with ZipFile(buffer, 'w') as zipObj:
            for audio in audios:
                zipObj.write("." + audio)

            zipObj.write(final_annotations_path)

        # remove temporary file
        if os.path.isfile(final_annotations_path):
            os.remove(final_annotations_path)

        file_response = HttpResponse(buffer.getvalue())
        file_response['Content-Type'] = 'application/x-zip-compressed'
        file_response['Content-Disposition'] = 'attachment; filename=audios.zip'

        return file_response
