import factory
from tasks.models import Task, get_review, Annotation, Comment
from users.models import User
from projects.models import Project, PredictionModels, Label
from django.contrib.auth.hashers import make_password
from django.test import TestCase


class UserFactory(factory.django.DjangoModelFactory):

    first_name = factory.Faker('first_name')
    last_name = factory.Faker('last_name')
    username = factory.Faker('email')
    password = factory.LazyFunction(lambda: make_password('pi3.1415'))

    class Meta:
        model = User


class TaskFactory(factory.django.DjangoModelFactory):

    name = factory.Faker('name')
    description = factory.Faker('text')
    user = factory.SubFactory(UserFactory)

    class Meta:
        model = Task


class AnnotationFactory(factory.django.DjangoModelFactory):

    task = factory.SubFactory(TaskFactory)
    user = factory.SubFactory(UserFactory)
    text = factory.Faker('text')

    class Meta:
        model = Annotation


class CommentFactory(factory.django.DjangoModelFactory):
    
        annotation = factory.SubFactory(AnnotationFactory)
        user = factory.SubFactory(UserFactory)
        text = factory.Faker('text')
    
        class Meta:
            model = Comment


class ProjectFactory(factory.django.DjangoModelFactory):

    name = factory.Faker('name')
    description = factory.Faker('text')
    user = factory.SubFactory(UserFactory)

    class Meta:
        model = Project


class PredictionModelFactory(factory.django.DjangoModelFactory):

    name = factory.Faker('name')
    description = factory.Faker('text')
    project = factory.SubFactory(ProjectFactory)

    class Meta:
        model = PredictionModels


class LabelFactory(factory.django.DjangoModelFactory):

    name = factory.Faker('name')
    description = factory.Faker('text')
    project = factory.SubFactory(ProjectFactory)

    class Meta:
        model = Label



