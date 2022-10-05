from django.test import TestCase, Client
from django.contrib.auth import get_user_model
from users.models import User


class TestIndexPage(TestCase):
    
    def test_index(self):
        request = self.client.get('/')
        self.assertEqual(request.status_code, 200)

    def test_invalid_page(self):
        # test page that does not exist
        request = self.client.get('/does_not_exist')
        self.assertEqual(request.status_code, 404)

class TestLoginPage(TestCase):

    def setUp(self):
        self.TestUser = get_user_model().objects.create_user(username='TestUserName', password='TestUserPassword', email='TestUserName@mail.com')
        self.TestUser.save()

    def test_login(self):

        c = Client()
        logged_in = c.login(username='TestUserName', password='TestUserPassword')
        self.assertEqual(logged_in, True)

    def test_wrong_login(self):
        c = Client()
        logged_in = c.login(username='TestUserName', password='TestUserWrongPassword')
        self.assertEqual(logged_in, False)
