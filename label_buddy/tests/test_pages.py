from django.test import TestCase, Client
from django.contrib.auth import get_user_model
from users.models import User
from django.test import tag


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

    def accounts_login(self):
        response = self.client.post('/accounts/login/', {'username': 'TestUserName', 'password': 'TestUserPassword'})
        self.assertTrue(response.data['authenticated'])

    def wrong_accounts_login(self):
        response = self.client.post('/accounts/login/', {'username': 'TestUserWrongName', 'password': 'TestUserPassword'})
        self.assertFalse(response.data['authenticated'])

class TestSignupPage(TestCase):

    def setUp(self):
        self.name = 'test user'
        self.email = 'testuser@email.com'
        self.username = 'testuser'
        self.password = 'password'

        self.user = {
            'name': self.name,
            'email': self.email,
            'username': self.username,
            'password': self.password
        }

    def accounts_signup(self):
        response = self.client.post('/accounts/signup/', self.user, format='text/html')
        self.assertEqual(response.status_code, 200)
