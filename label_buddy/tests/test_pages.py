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


# class TaskTest(TestCase):

#     def setUp(self):
        # self.TestUser = get_user_model().objects.create_user(username='TestUserName', password='TestUserPassword', email='TestUserName@mail.com')
        # self.TestUser.save()

#         list_of_users = [self.TestUser,]

#         self.TestLabel = Label.objects.create(name='TestLabel1')
#         self.TestLabel.save()

#         list_of_labels = [self.TestLabel,]

#         self.TestPredictionModel = PredictionModels.objects.create(title='TestPredictionModel')
#         self.TestPredictionModel_pk = self.TestPredictionModel.pk
#         self.TestPredictionModel.save()

        # list_of_predictionModels = [self.TestPredictionModel,]

        # self.TestProject = Project(title="TestProject", id=0)
        # self.TestPredictionModel_object = PredictionModels.objects.get(id=self.TestPredictionModel_pk)
        # print(self.TestPredictionModel_pk)
        # self.TestProject.prediction_model.add(self.TestPredictionModel_object)
        # self.TestProject.save()

        # self.TestProject.labels.add(self.TestLabel)
        # self.TestProject.reviewers.add(self.TestUser)
        # self.TestProject.labels.set([list_of_labels])
        # self.TestProject.reviewers.set([list_of_users])
        # self.TestProject.managers.set([list_of_users])
        # self.TestProject.annotators.set([list_of_users])

        # self.task = Task(project=self.TestProject, description="FirstDescription")
        # self.task.save()

    # def test_read_task(self):
    #     self.assertEqual(self.task.project, self.TestProject)

    # def test_update_task_description(self):
    #     self.TestPredictionModel.title = 'new description'
    #     self.TestPredictionModel.save()
    #     self.assertEqual(self.TestPredictionModel.title, 'new description')

    # def tearDown(self):
    #     self.TestProject.delete()

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