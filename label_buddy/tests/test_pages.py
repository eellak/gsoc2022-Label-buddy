from django.test import TestCase, Client
from django.contrib.auth import get_user_model


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