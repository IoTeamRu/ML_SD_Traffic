from celery import shared_task

from tlo import inference

@shared_task
def get_result():
    result = inference.run_model_on_trajectory()
    return result
