import os

from django.db import models
from django.db.models.signals import pre_save
from django.dispatch import receiver
from django.contrib.auth.models import AbstractUser

# Create your models here.
class User(AbstractUser):
    
    '''
    User class inherited from Django User model
    '''

    #remove unnecessary fields
    groups = None
    user_permissions = None

    #additional fields
    name = models.CharField(max_length=256, default="", db_index=True, help_text='Users full name')
    can_create_projects = models.BooleanField(default=False, help_text='True if the user can create projects (be a manager)')
    phone_number = models.CharField(max_length=256, blank=True, help_text="User's phone number")
    avatar = models.ImageField(upload_to='images', blank=True, help_text="User's avatar (image)")

    #How to display projects in admin
    def __str__(self):
        return '%s' % (self.username)

@receiver(pre_save, sender=User)
def auto_delete_file_on_change(sender, instance, **kwargs):
    """
    Deletes old file from filesystem
    when corresponding user object is updated
    with new file.
    """
    pk = instance.pk
    if not pk:
        return False

    try:
        old_avatar = User.objects.get(pk=pk).avatar
    except User.DoesNotExist:
        return False

    if old_avatar:
        new_avatar = instance.avatar
        if not old_avatar == new_avatar:
            if os.path.isfile(old_avatar.path):
                os.remove(old_avatar.path)