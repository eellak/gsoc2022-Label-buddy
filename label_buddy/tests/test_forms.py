from django.test import TestCase
from projects.forms import ProjectForm, PredictionModelForm
from tasks.forms import TaskForm
from users.forms import ExtendedLogInForm, ExtendedSignUpForm, ExtendedResetPasswordForm, UserForm


class ProjectFormTests(TestCase):

    def setUp(self):

        self.form = ProjectForm(data={
            "title": "a lowercase title",
            "description" : 'Testdescription',
            "instructions" : 'Testinstructions',
            "prediction_model" : 'Testprediction_model',
        })

    def test_prediction_model(self):
        
        self.assertEqual(
            self.form.errors["prediction_model"], ['Select a valid choice. That choice is not one of the available choices.']
        )

    def test_form(self):
        self.assertFalse(self.form.is_valid())


class PredictionModelFormTests(TestCase):

    def setUp(self):

        self.form = PredictionModelForm(data={
            "title": "a lowercase title",
            "output_labels" : 'Testoutput_labels',
            "image_repo" : 'Testimage_repo',
            "current_accuracy_precentage" : 'Testcurrent_accuracy_precentage',
            "current_loss_precentage" : 'Testcurrent_loss_precentage',
        })

    def test_current_accuracy_precentage(self):

        self.assertEqual(
            self.form.errors["current_accuracy_precentage"], ['Enter a number.']
        )

    def test_current_loss_precentage(self):

        self.assertEqual(
            self.form.errors["current_loss_precentage"], ['Enter a number.']
        )

    def test_form(self):
        self.assertFalse(self.form.is_valid())


class TaskFormTests(TestCase):

    def setUp(self):

        self.form = TaskForm(data={
            "file": "somefile.pdf",
        })

    def test_file(self):
        
        self.assertEqual(
            self.form.errors["file"], ['This field is required.', 'Even one of file or url should have a value.']
        )

    def test_form(self):
        self.assertFalse(self.form.is_valid())


class ExtendedLogInFormTests(TestCase):

    def setUp(self):

        self.form = ExtendedLogInForm(data={
            "login": "",
            "password": "",
        })

    def test_login(self):
        
        self.assertEqual(
            self.form.errors["login"], ['This field is required.']
        )

    def test_password(self):
        
        self.assertEqual(
            self.form.errors["login"], ['This field is required.']
        )

    def test_form(self):
        self.assertFalse(self.form.is_valid())


class ExtendedSignUpFormTests(TestCase):

    def setUp(self):

        self.form = ExtendedSignUpForm(data={
            "name": "",
            "email": "",
            "username" : "",
            "password1" : ""
        })

    def test_name(self):
        
        self.assertEqual(
            self.form.errors["name"], ['This field is required.']
        )
    
    def test_email(self):
        
        self.assertEqual(
            self.form.errors["email"], ['This field is required.']
        )

    def test_username(self):
        
        self.assertEqual(
            self.form.errors["username"], ['This field is required.']
        )

    def test_password1(self):
        
        self.assertEqual(
            self.form.errors["password1"], ['This field is required.']
        )

    def test_form(self):
        self.assertFalse(self.form.is_valid())


class ExtendedResetPasswordFormTests(TestCase):

    def setUp(self):

        self.form = ExtendedResetPasswordForm(data={
            "email": "",
        })

    def test_email(self):
        
        self.assertEqual(
            self.form.errors["email"], ['This field is required.']
        )
    
    def test_form(self):
        self.assertFalse(self.form.is_valid())


class ExtendedResetPasswordFormTests(TestCase):

    def setUp(self):

        self.form = ExtendedResetPasswordForm(data={
            "name" : "",
            "email" : "",
            "phone_number" : "",
            "avatar" : "",
        })
    
    def test_email(self):
        
        self.assertEqual(
            self.form.errors["email"], ['This field is required.']
        )

    def test_form(self):
        self.assertFalse(self.form.is_valid())