from django.db import models


class File(models.Model):
    name = models.CharField(max_length=255)
    content = models.TextField()
    status = models.CharField(max_length=20, choices=[('accepted', 'Accepted'), ('rejected', 'Rejected')], default='rejected')

    def __str__(self):
        return self.name

# Create your models here.
