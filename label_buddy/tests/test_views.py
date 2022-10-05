# from django.contrib.auth.models import AnonymousUser, User
from django.test import RequestFactory, TestCase
from django.test import TestCase, Client

from users.models import User

from projects.views import *
from users.views import UserList, edit_profile, UserDetail
from tasks.views import *

class UserViewsTest(TestCase):

    def setUp(self):
        # Every test needs access to the request factory.
        self.factory = RequestFactory()
        # self.user = User.objects.create_user(
        #     username='jacob', email='jacob@…', password='top_secret')

        self.TestUser = User.objects.create_user(username='TestUserName', password='TestUserPassword', email='TestUserName@mail.com')
        self.TestUser_pk = self.TestUser.pk
        self.TestUser.save()

    def test_UserList(self):
        # Create an instance of a GET request.
        request = self.factory.get('/users')

        # Recall that middleware are not supported. You can simulate a
        # logged-in user by setting request.user manually.
        request.user = self.TestUser

        # # Or you can simulate an anonymous user by setting request.user to
        # # an AnonymousUser instance.
        # request.user = AnonymousUser()

        # # Test my_view() as if it were deployed at /customer/details
        # response = my_view(request)

        # Use this syntax for class-based views.
        response = UserList.as_view()(request)

        self.assertEqual(response.status_code, 200)

    def test_edit_profile(self):

        request = self.factory.post('/user/TestUserName/edit')
        request.user = self.TestUser

        response = edit_profile(request, {'username' : 'TestUserName'})

        self.assertEqual(response.status_code, 302)

    def test_UserDetail(self):

        kwargs={'pk': f'{self.TestUser_pk}'}
        
        request = self.factory.get(f'/users/{self.TestUser_pk}')
        request.user = self.TestUser

        response = UserDetail.as_view()(request, **kwargs)

        self.assertEqual(response.status_code, 200)
