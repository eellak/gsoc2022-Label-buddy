from django.contrib.auth import authenticate, login
from projects.models import Project, Label, PredictionModels
from django.test import TestCase, Client
from django.contrib.auth import get_user_model
from tasks.models import Task
from django.contrib.auth.models import User
import factory


# class UserFactory(DjangoModelFactory):

#     username = factory.Sequence('testuser{}'.format)
#     email = factory.Sequence('testuser{}@company.com'.format)

#     class Meta:
#         model = get_user_model()


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


class PredictionModelTest(TestCase):

    def setUp(self):
        self.TestPredictionModel = PredictionModels.objects.create(title='TestPredictionModel')
        self.TestPredictionModel.save()

    def test_update_prediction_model_title(self):
        self.TestPredictionModel.title = 'new Title'
        self.TestPredictionModel.save()
        self.assertEqual(self.TestPredictionModel.title, 'new Title')

    def tearDown(self):
        self.TestPredictionModel.delete()


class LabelTest(TestCase):

    def setUp(self):
        self.TestLabel = Label.objects.create(name='TestLabel1')
        self.TestLabel.save()

    def tearDown(self):
        self.TestLabel.delete()


class UserTest(TestCase):

    def setUp(self):
        self.TestUser = get_user_model().objects.create_user(username='TestUserName', password='TestUserPassword', email='TestUserName@mail.com')
        self.TestUser.save()

    def test_update_user_username(self):
        self.TestUser.username = 'new TestUserName'
        self.TestUser.save()
        self.assertEqual(self.TestUser.username, 'new TestUserName')

    def tearDown(self):
        self.TestUser.delete()


class ProjectTest(TestCase):

    def setUp(self):

        self.TestPredictionModel = PredictionModels.objects.create(title='TestPredictionModel')
        self.TestPredictionModel.save()

        self.TestProject = Project(title="TestProject", prediction_model=self.TestPredictionModel)
        self.TestProject.save()
    
    def test_project_title(self):
        self.TestProject.title = 'new TestProject'
        self.TestProject.save()
        self.assertEqual(self.TestProject.title, 'new TestProject')

    def tearDown(self):
        self.TestPredictionModel.delete()
        self.TestProject.delete()


class TaskTest(TestCase):

    def setUp(self):
        self.TestUser = get_user_model().objects.create_user(username='TestUserName', password='TestUserPassword', email='TestUserName@mail.com')
        self.TestUser.save()

        self.TestLabel = Label.objects.create(name='TestLabel1')
        self.TestLabel.save()


        self.TestPredictionModel = PredictionModels.objects.create(title='TestPredictionModel')
        self.TestPredictionModel.save()


        self.TestProject = Project(title="TestProject", prediction_model=self.TestPredictionModel)
        self.TestProject.save()

        self.task = Task(project=self.TestProject)
        self.task.save()

    def test_read_task(self):
        self.assertEqual(self.task.project, self.TestProject)

    def tearDown(self):
        self.TestProject.delete()