from django.http import HttpResponse
from django.shortcuts import render
from . import tasks


def foo__(rquest):
    responce = tasks.get_result.delay()
    ready = responce.ready()
    result = responce.get()
    #return HttpResponse(f'<h1>The result is ready? {ready} / The result is {result}<h1>')
    return HttpResponse("home.html", {'result': result})


def trigger_task(request):
    if request.method == 'POST':
        arg1 = request.POST.get('arg1')  # Get data from the form
        responce = tasks.get_result.delay()
        result = responce.get()
        return render(request, 'home.html', {'result': result})
    return render(request, 'home.html')


def foo(request):
    if request.GET.get('launch'):
        responce = tasks.get_result.delay()
        result = responce.get()
        return render_to_response(request, 'home.html', {'result': result})


def home(request):
    return render(request, "home.html", {})
