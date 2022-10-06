from django import forms
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


class TaskFormTests(TestCase):

    def setUp(self):

        self.form = TaskForm(data={
            "file": "somefile.pdf",
        })

    def test_prediction_model(self):
        
        self.assertEqual(
            self.form.errors["file"], ['This field is required.', 'Even one of file or url should have a value.']
        )


# class ExtendedLogInFormTests(TestCase):

#     def setUp(self):

#         self.form = ExtendedLogInForm(data={
#             "file": "somefile.pdf",
#         })

#     def test_prediction_model(self):
        
#         self.assertEqual(
#             self.form.errors["file"], ['This field is required.', 'Even one of file or url should have a value.']
#         )