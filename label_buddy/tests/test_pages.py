from django.test import TestCase, Client
# from users.models import User
from projects.models import Project, Label, PredictionModels
from tasks.models import Task
import factory
from factory.django import DjangoModelFactory
from django.contrib.auth.models import User
from django.contrib.auth import get_user_model
from django.contrib.auth import authenticate, login


# class UserFactory(DjangoModelFactory):

#     username = factory.Sequence('testuser{}'.format)
#     email = factory.Sequence('testuser{}@company.com'.format)

#     class Meta:
#         model = get_user_model()


class TestIndexPage(TestCase):
    
    def test_index(self):
        request = self.client.get('/')
        self.assertEqual(request.status_code, 200)

    def test_invalid_page(self):
        # test page that does not exist
        request = self.client.get('/does_not_exist')
        self.assertEqual(request.status_code, 404)


class SigninTest(TestCase):

    def setUp(self):
        self.TestUser = get_user_model().objects.create_user(username='TestUserName', password='TestUserPassword', email='TestUserName@mail.com')
        self.TestUser.save()

    def tearDown(self):
        self.TestUser.delete()

    def test_correct(self):
        user = authenticate(username='TestUserName', password='TestUserPassword')
        self.assertTrue((user is not None) and user.is_authenticated)

    def test_wrong_username(self):
        user = authenticate(username='TestUserWrongName', password='TestUserPassword')
        self.assertFalse(user is not None and user.is_authenticated)

    def test_wrong_pssword(self):
        user = authenticate(username='TestUserName', password='TestUserWrongPassword')
        self.assertFalse(user is not None and user.is_authenticated)

class TaskTest(TestCase):

    def setUp(self):
        self.TestUser = get_user_model().objects.create_user(username='TestUserName', password='TestUserPassword', email='TestUserName@mail.com')
        self.TestUser.save()

        list_of_users = [self.TestUser,]

        self.TestLabel = Label.objects.create(name='TestLabel1')
        self.TestLabel.save()

        self.TestPredictionModel = PredictionModels.objects.create(title='TestPredictionModel', id=0)
        self.TestPredictionModel.save()

        list_of_labels = [self.TestLabel,]

        self.TestProject = Project(title="TestProject", id=0)
        self.TestProject.prediction_model.add(self.TestPredictionModel)

        # self.TestProject.labels.add(self.TestLabel)
        # self.TestProject.reviewers.add(self.TestUser)

        # self.TestProject.labels.set([list_of_labels])
        # self.TestProject.reviewers.set([list_of_users])
        # self.TestProject.managers.set([list_of_users])
        # self.TestProject.annotators.set([list_of_users])
        self.TestProject.save()

        self.task = Task(project=self.TestProject, description="FirstDescription")
        self.task.save()

    def tearDown(self):
        self.TestProject.delete()

    def test_read_task(self):
        self.assertEqual(self.task.project, self.TestProject)

    def test_update_task_description(self):
        self.task.description = 'new description'
        self.task.save()
        self.assertEqual(self.task.description, 'new description')



# c = Client()
# logged_in = c.login(username='TestUserName', password='TestUserPassword')
# self.assertEqual(logged_in, True)


# class TestCreatProjectPage(TestCase):

#     TestUser = None

#     def setUp(self):
#         """
#         Construct fake Users
#         :return:
#         """

#         self.TestUser = self.TestUser = User.objects.create(name='TestUserName', password='TestUserPassword', can_create_projects=True)


#     def test_create_project(self):

#         c = Client()
#         response = c.post('/projects/create', {'username': 'TestUserName', 'password': 'TestUserPassword'})
#         self.assertEqual(response.status_code, 200)

#     def test_invalid_page(self):
#         # test page that does not exist
#         c = Client()
#         response = c.post('/projects/create/does_not_exist', {'username': 'TestUserName', 'password': 'TestUserPassword'})
#         self.assertEqual(response.status_code, 404)