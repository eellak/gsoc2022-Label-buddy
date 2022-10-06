# from django.contrib.auth.models import AnonymousUser, User
from multiprocessing import managers
from django.test import RequestFactory, TestCase, Client
from django.urls import reverse

from users.models import User
from projects.models import Project, PredictionModels
from tasks.models import Task

from users.forms import UserForm

from projects.views import index, project_create_view, project_delete_view, project_edit_view, project_add_prediction_model_view, project_page_view, model_page_view, ProjectList, ProjectDetail, ProjectTasks, api_root, get_dataset_view, project_add_prediction_model_view, annotation_delete_view, task_delete_view, annotate_task_view, list_annotations_for_task_view, review_annotation_view, AnnotationPredictions, get_user
from users.views import UserList, edit_profile, UserDetail
from tasks.views import TaskList, AnnotationSave, ExportData, ExportDataToContainer

class UserViewsTest(TestCase):

    def setUp(self):
        # Every test needs access to the request factory.
        self.factory = RequestFactory()
        # self.user = User.objects.create_user(
        #     username='jacob', email='jacob@…', password='top_secret')

        self.TestUser = User.objects.create_user(username='TestUserName', password='TestUserPassword', email='TestUserName@mail.com')
        self.TestUser_pk = self.TestUser.pk
        self.TestUser.save()

    def test_UserList(self):
        # Create an instance of a GET request.
        request = self.factory.get('/users')

        # Recall that middleware are not supported. You can simulate a
        # logged-in user by setting request.user manually.
        request.user = self.TestUser

        # # Or you can simulate an anonymous user by setting request.user to
        # # an AnonymousUser instance.
        # request.user = AnonymousUser()

        # # Test my_view() as if it were deployed at /customer/details
        # response = my_view(request)

        # Use this syntax for class-based views.
        response = UserList.as_view()(request)

        self.assertEqual(response.status_code, 200)

    def test_edit_profile(self):

        kwargs= {'username' : 'TestUserName'}

        # user_form = UserForm()

        request = self.factory.post('/user/TestUserName/edit')
        request.user = self.TestUser

        response = edit_profile(request, **kwargs)

        self.assertEqual(response.status_code, 200)

    def test_wrong_edit_profile(self):

        kwargs= {'username' : 'TestUserWrongName'}

        request = self.factory.post('/user/TestUserWrongName/edit')
        request.user = self.TestUser

        response = edit_profile(request, **kwargs)

        self.assertEqual(response.status_code, 302)

    def test_UserDetail(self):

        kwargs={'pk': f'{self.TestUser_pk}'}

        request = self.factory.get(f'/users/{self.TestUser_pk}')
        request.user = self.TestUser

        response = UserDetail.as_view()(request, **kwargs)

        self.assertEqual(response.status_code, 200)

    def test_get_user(self):
        user = get_user(username="TestUserName") 
        self.assertEqual(self.TestUser, user)




class ProjectViewsTest(TestCase):

    def setUp(self):
        # Every test needs access to the request factory.
        self.factory = RequestFactory()

        self.TestUser = User.objects.create_user(username='TestUserName', password='TestUserPassword', 
        email='TestUserName@mail.com')
        self.TestUser.can_create_projects = True
        self.TestUser.save()
        self.TestUser_pk = self.TestUser.pk
        self.TestUser.save()

        self.TestPredictionModel = PredictionModels.objects.create(title='TestPredictionModel')
        self.TestPredictionModel.save()

        self.TestProject = Project(title="TestProject", prediction_model=self.TestPredictionModel, id=1)
        self.TestProject.managers.add(self.TestUser)
        # self.TestProject.reviwers.add(self.TestUser)
        self.TestProject.annotators.add(self.TestUser)
        self.TestProject.save()

        # self.task = Task(project=self.TestProject)
        # self.task.save()
    
    def test_index(self):

        request = self.factory.post('/')
        request.user = self.TestUser

        response = index(request)

        self.assertEqual(response.status_code, 200)

    # def test_project_create_view(self):

    #     self.TestUser.can_create_projects = True
    #     self.TestUser.save()

    #     # kwargs = {"title": "TestProject", "prediction_model" : f'{self.TestPredictionModel}'}
    #     request = self.factory.post('/projects/create')
    #     request.user = self.TestUser

    #     response = project_create_view(request)

    #     self.assertEqual(response.status_code, 200)

    # def test_project_add_prediction_model_view(self):

    #     request = self.factory.post('projects/add_prediction_model')
    #     request.user = self.TestUser

    #     response = project_add_prediction_model_view(request)

    #     self.assertEqual(response.status_code, 200)

    def test_project_edit_view(self):

        kwargs={'pk': '1'}

        request = self.factory.get(f'projects/1/edit')
        request.user = self.TestUser

        response = project_edit_view(request, **kwargs)

        self.assertEqual(response.status_code, 200)

    def test_project_delete_view(self):

        kwargs={'pk': '1'}

        request = self.factory.get(f'projects/1/delete')
        request.user = self.TestUser

        response = project_delete_view(request, **kwargs)

        self.assertEqual(response.status_code, 200)

    # # message problem 
    # def test_annotation_delete_view(self):

    #     kwargs={'pk': '1', 'task_pk' : '1'}

    #     request = self.factory.post(f'projects/1/tasks/1/annotation/delete')
    #     request.user = self.TestUser

    #     response = annotation_delete_view(request, **kwargs)

    #     self.assertEqual(response.status_code, 200)

    # # message problem 
    # def test_task_delete_view(self):

    #     kwargs={'pk': '1', 'task_pk' : '1'}

    #     request = self.factory.post(f'projects/1/tasks/delete')
    #     request.user = self.TestUser

    #     response = task_delete_view(request, **kwargs)

    #     self.assertEqual(response.status_code, 200)

    def test_project_page_view(self):
        kwargs={'pk': '1'}

        request = self.factory.get(f'projects/1/tasks')
        request.user = self.TestUser

        response = project_page_view(request, **kwargs)

        self.assertEqual(response.status_code, 200)

    def test_model_page_view(self):
        kwargs={'pk': '1'}

        request = self.factory.get(f'projects/1/tasks/model')
        request.user = self.TestUser

        response = model_page_view(request, **kwargs)

        self.assertEqual(response.status_code, 200)

    # # message problem
    # def test_annotate_task_view(self):

    #     kwargs={'pk': '1', 'task_pk' : '1'}

    #     request = self.factory.post(f'projects/1/tasks/1/annotation')
    #     request.user = self.TestUser

    #     response = annotate_task_view(request, **kwargs)

    #     self.assertEqual(response.status_code, 200)

    # # messages problem
    # def test_list_annotations_for_task_view(self):

    #     kwargs={'pk': '1', 'task_pk' : '1'}

    #     request = self.factory.post(f'projects/1/tasks/1/list_annotation')
    #     request.user = self.TestUser

    #     response = list_annotations_for_task_view(request, **kwargs)

    #     self.assertEqual(response.status_code, 200)

    # # message problem
    # def test_review_annotation_view(self):

    #     kwargs={'pk': '1', 'task_pk' : '1', 'annotation_pk': '1'}

    #     request = self.factory.post(f'projects/1/tasks/1/list_annotation/1/review')
    #     request.user = self.TestUser

    #     response = review_annotation_view(request, **kwargs)

    #     self.assertEqual(response.status_code, 200)

    def test_ProjectList(self):

        request = self.factory.get(f'projects')
        request.user = self.TestUser

        response = ProjectList.as_view()(request)

        self.assertEqual(response.status_code, 200)

    def test_ProjectDetail(self):
        kwargs={'pk': '1'}

        request = self.factory.get(f'projects/1')
        request.user = self.TestUser

        response = ProjectDetail.as_view()(request, **kwargs)

        self.assertEqual(response.status_code, 200)
  
    def test_ProjectTasks(self):
        kwargs={'pk': '1'}

        request = self.factory.get(f'projects/1/tasks')
        request.user = self.TestUser

        response = ProjectTasks.as_view()(request, **kwargs)

        self.assertEqual(response.status_code, 200)

    def test_api_root(self):
        request = self.factory.get(f'/root')
        request.user = self.TestUser

        response = api_root(request)

        self.assertEqual(response.status_code, 200)

    def test_AnnotationPredictions(self):

        kwargs={'pk': '1', 'task_pk' : '1'}

        request = self.factory.post(f'/projects/1/tasks/1/annotation/predict')
        request.user = self.TestUser

        response = AnnotationPredictions.as_view()(request, **kwargs)

        self.assertEqual(response.status_code, 403)
    
    # def test_get_dataset_view(self):

    #     kwargs={'data': 'training'}

    #     request = self.factory.post(f'/projects/get_dataset')
    #     request.user = self.TestUser

    #     print(request)

    #     response = get_dataset_view.as_view()(request, **kwargs)

    #     print(response)

    #     self.assertEqual(response.status_code, 200)


class TaskViewsTest(TestCase):

    def setUp(self):
        # Every test needs access to the request factory.
        self.factory = RequestFactory()

        self.TestUser = User.objects.create_user(username='TestUserName', password='TestUserPassword', 
        email='TestUserName@mail.com')
        self.TestUser_pk = self.TestUser.pk
        self.TestUser.can_create_projects = True
        self.TestUser.save()


        self.TestPredictionModel = PredictionModels.objects.create(title='TestPredictionModel')
        self.TestPredictionModel.save()

        self.TestProject = Project(title="TestProject", prediction_model=self.TestPredictionModel, id=1)
        # self.TestProject.managers.add(self.TestUser)
        # self.TestProject.reviwers.add(self.TestUser)
        self.TestProject.annotators.add(self.TestUser)
        self.TestProject.save()

        # self.task = Task(project=self.TestProject)
        # self.task.save()

    def test_TaskList(self):
        request = self.factory.get(reverse('task-list'))
        request.user = self.TestUser

        response = TaskList.as_view()(request)

        self.assertEqual(response.status_code, 200)

    def test_AnnotationSave(self):

        kwargs={'pk': '1', 'task_pk' : '1'}

        request = self.factory.post('/projects/1/tasks/1/annotation/save')
        request.user = self.TestUser

        response = AnnotationSave.as_view()(request, **kwargs)

        self.assertEqual(response.status_code, 403)

    def test_ExportData(self):

        kwargs={'pk': '1'}

        request = self.factory.post('/projects/1/tasks/export')
        request.user = self.TestUser

        response = ExportData.as_view()(request, **kwargs)

        self.assertEqual(response.status_code, 403)

    def test_ExportDataToContainer(self):

        kwargs={'pk': '1'}

        request = self.factory.post('/projects/1/tasks/export_to_container')
        request.user = self.TestUser

        response = ExportDataToContainer.as_view()(request, **kwargs)

        self.assertEqual(response.status_code, 403)