from users.models import User
from .models import (
    Annotation,
    Task,
    Status,
    Comment,
    Annotation_status,
)


def get_user(username):

    """
    Get user by username.
    """

    try:
        user = User.objects.get(username=username)
        return user
    except User.DoesNotExist:
        return None


def get_annotation(task, project, user):

    """
    Get annotation by task, project and user.
    """

    try:
        annotation = Annotation.objects.get(task=task, project=project, user=user)
        return annotation
    except Annotation.DoesNotExist:
        return None


def get_review(annotation):

    """
    Get annotation's review (if exists).
    """

    try:
        review = Comment.objects.get(annotation=annotation)
        return review
    except Comment.DoesNotExist:
        return None


def export_data(project, export_only_approved):

    """
    For all tasks of project which have been annotated result will be an array of dicts. Each dict will represent a task
    which will contain all annotation completed for this task.
    """

    exported_result = []
    skipped_annotations = 0
    # Get all annotated tasks of project
    annotated_tasks = Task.objects.filter(project=project, status=Status.labeled)

    audios = []

    for task in annotated_tasks:
        task_dict = {
            "id": task.id,
            "annotations": [],
            "file_upload": task.original_file_name,
            "data": {
                "audio": task.file.url,
            },
            "project_created_at": project.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "project": project.id,
        }

        audios.append(task.file.url)

        task_annotations = Annotation.objects.filter(task=task, project=project)

        # For every annotation, push it
        for annotation in task_annotations:
            if export_only_approved:
                if annotation.review_status == Annotation_status.approved:
                    annotation_user = annotation.user
                    review = get_review(annotation)
                    assert review is not None, "Annotation approved but not reviewed"
                    annotation_dict = {
                        "id": annotation.id,
                        "completed_by": {
                            "id": annotation_user.id,
                            "username": annotation_user.username,
                            "email": annotation_user.email,
                            "name": annotation_user.name,
                        },
                        "reviewed_by": {} if not review else {
                            "id": review.reviewed_by.id,
                            "username": review.reviewed_by.username,
                            "email": review.reviewed_by.email,
                            "name": review.reviewed_by.name,
                            "review_status": annotation.review_status.name,
                            "review_created_at": review.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                            "review_updated_at": review.updated_at.strftime("%Y-%m-%d %H:%M:%S") if review.updated_at else "",
                        },
                        "result": annotation.result,
                        "created_at": annotation.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "updated_at": annotation.updated_at.strftime("%Y-%m-%d %H:%M:%S") if annotation.updated_at else "",
                        "task": task.id,
                    }
                    task_dict["annotations"].append(annotation_dict)
                else:
                    skipped_annotations += 1
            else:
                annotation_user = annotation.user
                review = get_review(annotation)
                annotation_dict = {
                    "id": annotation.id,
                    "completed_by": {
                        "id": annotation_user.id,
                        "username": annotation_user.username,
                        "email": annotation_user.email,
                        "name": annotation_user.name,
                    },
                    "reviewed_by": {} if not review else {
                        "id": review.reviewed_by.id,
                        "username": review.reviewed_by.username,
                        "email": review.reviewed_by.email,
                        "name": review.reviewed_by.name,
                        "review_status": annotation.review_status.name,
                        "review_created_at": review.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "review_updated_at": review.updated_at.strftime("%Y-%m-%d %H:%M:%S") if review.updated_at else "",
                    },
                    "result": annotation.result,
                    "created_at": annotation.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "updated_at": annotation.updated_at.strftime("%Y-%m-%d %H:%M:%S") if annotation.updated_at else "",
                    "task": task.id,
                }
                task_dict["annotations"].append(annotation_dict)

        exported_result.append(task_dict)
    return exported_result, skipped_annotations, audios


def format_exported_json(exported_json):

    final_annotations = {}
    for audio_id in exported_json:
        audio_name = audio_id['data']['audio'].split('audio/')[1].split('.')[0]
        formated_annotations = []
        annotation_results = audio_id['annotations'][0]['result']
        for annotation_result in annotation_results:
            start = annotation_result['value']['start']
            end = annotation_result['value']['end']
            label = annotation_result['value']['label']
            formated_annotation = [start, end, label]
            formated_annotations.append(formated_annotation)
        
        final_annotations[audio_name] = formated_annotations

    return final_annotations
