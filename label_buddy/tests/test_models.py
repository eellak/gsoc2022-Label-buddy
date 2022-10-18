from django.contrib.auth import authenticate
from projects.models import Project, Label, PredictionModels
from django.test import TestCase, TransactionTestCase
from tasks.models import Task, get_review, Annotation, Comment
from users.models import User
from users.views import get_user
from django.core.exceptions import ValidationError 
from django.dispatch import Signal,receiver


class SigninTest(TestCase):

    def setUp(self):
        self.TestUser = User.objects.create_user(username='TestUserName', password='TestUserPassword', email='TestUserName@mail.com')
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

    def strTest(self):
        self.assertFalse(str(self.TestPredictionModel), 'TestPredictionModel')


class LabelTest(TestCase):

    def setUp(self):
        self.TestLabel = Label.objects.create(name='TestLabel1')
        self.TestLabel.save()

    def tearDown(self):
        self.TestLabel.delete()

    def test_name(self):
        self.assertEqual(str(self.TestLabel), 'TestLabel1')

    def test_str(self):
        self.assertEqual(str(self.TestLabel), 'TestLabel1')


class UserTest(TestCase):

    def setUp(self):
        self.TestUser = User.objects.create_user(username='TestUserName', password='TestUserPassword', 
        email='TestUserName@mail.com')
        self.TestUser.can_create_projects = False
        self.TestUser.save()

    def test_update_user_username(self):
        self.TestUser.username = 'new TestUserName'
        self.TestUser.save()
        self.assertEqual(self.TestUser.username, 'new TestUserName')

    def test_str(self):
        self.assertEqual(str(self.TestUser), 'TestUserName')

    def test_can_create_projects(self):
        self.TestUser.is_superuser = True
        self.TestUser.save()
        self.assertEqual(self.TestUser.can_create_projects, True)

    def tearDown(self):
        self.TestUser.delete()
        user = get_user(username="TestUserName") 
        self.assertEqual(user, None)


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
    
    def test_str(self):
        self.assertEqual(str(self.TestProject), 'TestProject')


class TaskTest(TestCase):

    def setUp(self):
        self.TestUser = User.objects.create_user(username='TestUserName', password='TestUserPassword', email='TestUserName@mail.com')
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

    def test_str(self):
        self.assertEqual(str(self.task), 'Task: 1 - project: TestProject')

    def cleanTest(self):
        self.assertRaises(ValidationError, self.task.clean())


class AnnotationTest(TestCase):

    def setUp(self):
        self.TestUser = User.objects.create_user(username='TestUserName', password='TestUserPassword', email='TestUserName@mail.com')
        self.TestUser.save()

        self.TestLabel = Label.objects.create(name='TestLabel1')
        self.TestLabel.save()

        self.TestPredictionModel = PredictionModels.objects.create(title='TestPredictionModel')
        self.TestPredictionModel.save()

        self.TestProject = Project(title="TestProject", prediction_model=self.TestPredictionModel)
        self.TestProject.save()

        self.task = Task(project=self.TestProject)
        self.task.save()

        self.TestAnnotation = Annotation(task=self.task, project=self.TestProject, user=self.TestUser)
        self.TestAnnotation.save()

        self.TestComment = Comment(reviewed_by=self.TestUser, annotation=self.TestAnnotation, comment='TestComment')
        self.TestComment.save()

    def test_get_review(self):
        review = get_review(annotation=self.TestAnnotation)
        self.assertEqual(str(review), 'Comment from TestUserName')

    def test_deleted_get_review(self):
        self.TestComment.delete()
        review = get_review(annotation=self.TestAnnotation)
        self.assertEqual(review, None)

    def test_str(self):
        self.assertEqual(str(self.TestAnnotation), 'Annotation 1 - project: TestProject')


class CommentTest(TestCase):

    def setUp(self):
        self.TestUser = User.objects.create_user(username='TestUserName', password='TestUserPassword', email='TestUserName@mail.com')
        self.TestUser.save()

        self.TestLabel = Label.objects.create(name='TestLabel1')
        self.TestLabel.save()

        self.TestPredictionModel = PredictionModels.objects.create(title='TestPredictionModel')
        self.TestPredictionModel.save()

        self.TestProject = Project(title="TestProject", prediction_model=self.TestPredictionModel)
        self.TestProject.save()

        self.task = Task(project=self.TestProject)
        self.task.save()

        self.TestAnnotation = Annotation(task=self.task, project=self.TestProject, user=self.TestUser)
        self.TestAnnotation.save()

        self.TestComment = Comment(reviewed_by=self.TestUser, annotation=self.TestAnnotation, comment='TestComment')
        self.TestComment.save()

    def test_str(self):
        self.assertEqual(str(self.TestComment), 'Comment from TestUserName')


class TestsThatDependsOnPrimaryKeySequences(TransactionTestCase):
    reset_sequences = True

    def test_user_pk(self):
        self.TestUser = User.objects.create_user(username='TestUserName', password='TestUserPassword', email='TestUserName@mail.com')
        self.TestUser.save()
        # lion.pk is guaranteed to always be 1
        self.assertEqual(self.TestUser.pk, 1)

    def test_prediction_model_pk(self):
        self.TestPredictionModel = PredictionModels.objects.create(title='TestPredictionModel')
        self.TestPredictionModel.save()
        # lion.pk is guaranteed to always be 1
        self.assertEqual(self.TestPredictionModel.pk, 1)

    def test_label_pk(self):
        self.TestLabel = Label.objects.create(name='TestLabel')
        self.TestLabel.save()
        # lion.pk is guaranteed to always be 1
        self.assertEqual(self.TestLabel.pk, 'TestLabel')

    def test_project_pk(self):
        self.TestPredictionModel = PredictionModels.objects.create(title='TestPredictionModel')
        self.TestPredictionModel.save()

        self.TestProject = Project(title="TestProject", prediction_model=self.TestPredictionModel)
        self.TestProject.save()
        # lion.pk is guaranteed to always be 1
        self.assertEqual(self.TestProject.pk, 1)

    def test_task_pk(self):
        self.TestUser = User.objects.create_user(username='TestUserName', password='TestUserPassword', email='TestUserName@mail.com')
        self.TestUser.save()

        self.TestLabel = Label.objects.create(name='TestLabel1')
        self.TestLabel.save()

        self.TestPredictionModel = PredictionModels.objects.create(title='TestPredictionModel')
        self.TestPredictionModel.save()

        self.TestProject = Project(title="TestProject", prediction_model=self.TestPredictionModel)
        self.TestProject.save()

        self.TestTask = Task(project=self.TestProject)
        self.TestTask.save()
        # lion.pk is guaranteed to always be 1
        self.assertEqual(self.TestTask.pk, 1)

    
    def test_label_pk(self):
        self.TestUser = User.objects.create_user(username='TestUserName', password='TestUserPassword', email='TestUserName@mail.com')
        self.TestUser.save()

        self.TestLabel = Label.objects.create(name='TestLabel1')
        self.TestLabel.save()

        self.TestPredictionModel = PredictionModels.objects.create(title='TestPredictionModel')
        self.TestPredictionModel.save()

        self.TestProject = Project(title="TestProject", prediction_model=self.TestPredictionModel)
        self.TestProject.save()

        self.task = Task(project=self.TestProject)
        self.task.save()

        self.TestAnnotation = Annotation(task=self.task, project=self.TestProject, user=self.TestUser)
        self.TestAnnotation.save()

        # lion.pk is guaranteed to always be 1
        self.assertEqual(self.TestAnnotation.pk, 1)

    def test_prediction_model_pk(self):
        self.TestUser = User.objects.create_user(username='TestUserName', password='TestUserPassword', email='TestUserName@mail.com')
        self.TestUser.save()

        self.TestLabel = Label.objects.create(name='TestLabel1')
        self.TestLabel.save()

        self.TestPredictionModel = PredictionModels.objects.create(title='TestPredictionModel')
        self.TestPredictionModel.save()

        self.TestProject = Project(title="TestProject", prediction_model=self.TestPredictionModel)
        self.TestProject.save()

        self.task = Task(project=self.TestProject)
        self.task.save()

        self.TestAnnotation = Annotation(task=self.task, project=self.TestProject, user=self.TestUser)
        self.TestAnnotation.save()

        self.TestComment = Comment(reviewed_by=self.TestUser, annotation=self.TestAnnotation, comment='TestComment')
        self.TestComment.save()

        # lion.pk is guaranteed to always be 1
        self.assertEqual(self.TestComment.pk, 1)


    
    

    
    

