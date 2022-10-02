from django.test import TestCase, Client
from users.models import User

TestUser = User.objects.create(name='TestUser', password='testuserpassword', can_create_projects=True)

class TestIndexPage(TestCase):
    def test_index(self):
        request = self.client.get('/')
        self.assertEqual(request.status_code, 200)

    def test_invalid_page(self):
        # test page that does not exist
        request = self.client.get('/does_not_exist')
        self.assertEqual(request.status_code, 404)


class TestCreatProjectPage(TestCase):
    def test_create_project(self):

        c = Client()
        response = c.post('/projects/create', {'username': 'TestUserName', 'password': 'testuserpassword'})
        self.assertEqual(response.status_code, 200)

    def test_invalid_page(self):
        # test page that does not exist
        c = Client()
        response = c.post('/projects/create/does_not_exist', {'username': 'TestUserName', 'password': 'testuserpassword'})
        self.assertEqual(response.status_code, 404)