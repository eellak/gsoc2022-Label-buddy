from django.test import TestCase
from .models import Project, User

class ProjectTestCase(TestCase):
    def setUp(self):

        TestUser = User.objects.create(name='TestUser', email='testuser@test.com', password='top_secret', can_create_projects=True, phone_number="6969696969")
        Project.objects.create(title="TestProject", description="TestDiscription", instructions="TestInstructions",
        labels="TestLabels",  reviewers=TestUser, annotators=TestUser, managers=TestUser)
        Project.objects.create(name="cat", sound="meow")
