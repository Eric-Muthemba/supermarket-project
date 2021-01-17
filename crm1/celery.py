import os

from celery import Celery
from celery.schedules import crontab

from django.conf import settings
# set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'crm1.settings')

app = Celery('crm1')

app.config_from_object('django.conf:settings', namespace='CELERY')

app.conf.beat_schedule = {'every-15-seconds':
                              { 'task':'accounts.analysis.recommend_products.recommend',
                                        'schedule': 3600,
                                        #'schedule': crontab(hour=16, day_of_week=5),
                                        'args':("1",),
                              }
                         }

app.autodiscover_tasks(lambda :settings.INSTALLED_APPS)


@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')




#celery -A crm1 worker -B -l info